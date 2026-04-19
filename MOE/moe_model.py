"""
moe_model.py — Clase principal MoE
Pipeline completo: imagen/volumen → backbone → router → experto → predicción

Flujo:
  x  →  AdaptivePreprocessor  →  SharedBackbone (CLS token)
      →  KNNRouter (expert_id) →  Expert_k (logits/probs)  →  output
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Union, Dict, Optional, Tuple, Any

from .preprocess  import AdaptivePreprocessor
from .backbone    import SharedBackbone, build_backbone
from .router_knn  import KNNRouter, build_router
from .experts     import load_all_experts, EXPERT_META


class MoEModel(nn.Module):
    """
    Mixture of Experts médico multimodal.

    Parámetros
    ----------
    base_dir        : raíz del proyecto (contiene expertos/ y embedings/)
    device          : "cpu" o "cuda"
    use_fp16        : activar inferencia en float16 (solo GPU)
    backbone_name   : modelo timm para el backbone
    knn_k           : vecinos del router
    n_slices_3d     : slices 3D para el backbone
    embeddings_dir  : directorio con all_train_Z.npy y all_train_expert_y.npy
    """

    def __init__(
        self,
        base_dir:       Union[str, Path] = ".",
        device:         str  = "cpu",
        use_fp16:       bool = False,
        backbone_name:  str  = "vit_tiny_patch16_224",
        knn_k:          int  = 5,
        n_slices_3d:    int  = 8,
        embeddings_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__()

        self.device   = torch.device(device)
        self.use_fp16 = use_fp16 and ("cuda" in device)
        self.base_dir = Path(base_dir)

        emb_dir = Path(embeddings_dir) if embeddings_dir else \
                  self.base_dir / "embedings" / "Output embeddings"

        # ── Componentes ───────────────────────────────────────────────────
        print("\n[MoE] Iniciando sistema...")

        self.preprocessor = AdaptivePreprocessor(device=device)

        self.backbone = build_backbone(
            model_name  = backbone_name,
            pretrained  = True,
            device      = device,
            n_slices_3d = n_slices_3d,
        )

        self.router = build_router(
            embeddings_dir = emb_dir,
            k              = knn_k,
            device         = device,
        )

        # Cargar expertos como nn.ModuleDict para que se registren correctamente
        experts_dict = load_all_experts(base_dir=base_dir, device=device)
        self.experts = nn.ModuleDict({str(k): v for k, v in experts_dict.items()})

        if self.use_fp16:
            self.backbone.half()
            for exp in self.experts.values():
                exp.half()
            print("[MoE] FP16 activado")

        print("[MoE] Sistema listo.\n")

    # ── Inferencia desde tensor ───────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, dict]:
        """
        Parámetros
        ----------
        x : tensor crudo (ya cargado), rank 4 (2D) o 5 (3D)

        Retorna
        -------
        output    : logits / probabilidades del experto seleccionado
        expert_id : ID del experto (0-4)
        router_info : dict con detalles del routing
        """
        # 1. Preprocesar
        x_proc, modality = self.preprocessor.from_tensor(x)

        # 2. Backbone → CLS embedding
        if self.use_fp16:
            x_proc = x_proc.half()
        z = self.backbone(x_proc)  # (B, d_model)

        # 3. Router → expert_id (toma primer elemento del batch para routing)
        z_np       = z[0].float().cpu().numpy()
        router_info = self.router.predict_with_scores(z_np)
        expert_id   = router_info["expert_id"]

        # 4. Experto seleccionado
        expert = self.experts[str(expert_id)]

        # Preparar input para el experto
        expert_input = self._prepare_expert_input(x_proc, expert_id, modality)
        if self.use_fp16:
            expert_input = expert_input.half()

        output = expert(expert_input)  # logits (B, num_classes)

        # 5. Post-procesado según tarea
        output = self._postprocess(output, expert_id)

        router_info["modality"] = modality
        router_info["expert_name"] = EXPERT_META[expert_id]["name"]
        return output, expert_id, router_info

    # ── Inferencia desde archivo ──────────────────────────────────────────

    @torch.no_grad()
    def predict_from_file(
        self,
        path: Union[str, Path],
    ) -> Tuple[torch.Tensor, int, dict]:
        """
        Carga desde archivo y ejecuta el pipeline completo.
        Equivalente a forward() pero acepta rutas de archivo.
        """
        x_proc, modality = self.preprocessor.from_file(path)

        if self.use_fp16:
            x_proc = x_proc.half()

        z = self.backbone(x_proc)
        z_np = z[0].float().cpu().numpy()
        router_info = self.router.predict_with_scores(z_np)
        expert_id   = router_info["expert_id"]

        expert       = self.experts[str(expert_id)]
        expert_input = self._prepare_expert_input(x_proc, expert_id, modality)
        if self.use_fp16:
            expert_input = expert_input.half()

        output = expert(expert_input)
        output = self._postprocess(output, expert_id)

        router_info["modality"] = modality
        router_info["expert_name"] = EXPERT_META[expert_id]["name"]
        return output, expert_id, router_info

    # ── Helpers internos ──────────────────────────────────────────────────

    def _prepare_expert_input(
        self,
        x_proc:    torch.Tensor,
        expert_id: int,
        modality:  str,
    ) -> torch.Tensor:
        """
        Adapta el tensor preprocesado al formato esperado por cada experto.
        - Expertos 2D (0,1,2): necesitan (B, 3, 224, 224)
        - Expertos 3D (3,4):   necesitan (B, 1, 64, 64, 64)
        El preprocesador ya produce el shape correcto; solo verificamos.
        """
        if expert_id in (0, 1, 2):
            # 2D experts — x_proc debería ser (B, 3, 224, 224)
            if x_proc.dim() == 5:
                # Volumen llegó a experto 2D: tomamos slice central
                D = x_proc.shape[2]
                sl = x_proc[:, :, D // 2, :, :]  # (B, 1, H, W)
                if sl.shape[1] == 1:
                    sl = sl.repeat(1, 3, 1, 1)
                import torch.nn.functional as F
                sl = F.interpolate(sl.float(), size=(224, 224),
                                   mode="bilinear", align_corners=False)
                return sl
            return x_proc

        else:
            # 3D experts — x_proc debería ser (B, 1, 64, 64, 64)
            if x_proc.dim() == 4:
                # Imagen 2D llegó a experto 3D: extender dimensión Z
                # (B, 3, 224, 224) → (B, 1, 64, 64, 64) via promedio de canales + repeat
                import torch.nn.functional as F
                x2d = x_proc[:, :1, :, :]  # tomar solo 1 canal
                x2d = F.interpolate(x2d.float(), size=(64, 64),
                                    mode="bilinear", align_corners=False)
                # Añadir eje Z replicando el slice 64 veces
                x3d = x2d.unsqueeze(2).repeat(1, 1, 64, 1, 1)
                return x3d
            return x_proc

    def _postprocess(self, output: torch.Tensor, expert_id: int) -> torch.Tensor:
        """Aplica activación según tarea del experto."""
        task = EXPERT_META[expert_id]["task"]
        output = output.float()
        if task == "multilabel":
            return torch.sigmoid(output)
        elif task in ("multiclass", "binary"):
            return torch.softmax(output, dim=-1)
        return output

    # ── Utilidades ────────────────────────────────────────────────────────

    def get_expert_info(self, expert_id: int) -> dict:
        return EXPERT_META[expert_id]

    def to(self, device, **kwargs):
        self.device = torch.device(device)
        self.preprocessor.device = self.device
        self.backbone.device     = self.device
        return super().to(device, **kwargs)
