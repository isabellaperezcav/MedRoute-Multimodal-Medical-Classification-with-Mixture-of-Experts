"""
backbone.py — Backbone ViT compartido (congelado)
Usa timm para cargar ViT-Tiny preentrenado y extraer el CLS token (dim=192).
El backbone es 2D: para volúmenes 3D se aplica slice-pooling sobre el eje Z.
"""

import torch
import torch.nn as nn
from typing import Optional


# ── Backbone ViT ──────────────────────────────────────────────────────────────

class SharedBackbone(nn.Module):
    """
    Backbone ViT-Tiny de timm con pesos preentrenados (congelado).
    Extrae el CLS token (d_model=192) para 2D/3D.

    Para volúmenes 3D (B, 1, D, H, W):
      - Extrae N slices del centro del volumen
      - Pasa cada slice por el ViT
      - Promedia los CLS tokens → embedding final

    Parámetros
    ----------
    model_name  : nombre timm (default: vit_tiny_patch16_224)
    pretrained  : cargar pesos ImageNet
    freeze      : congelar todos los parámetros
    device      : dispositivo destino
    n_slices_3d : número de slices centrales para volúmenes 3D
    """

    def __init__(
        self,
        model_name:  str  = "vit_tiny_patch16_224",
        pretrained:  bool = True,
        freeze:      bool = True,
        device:      str  = "cpu",
        n_slices_3d: int  = 8,
    ):
        super().__init__()
        import timm

        self.device      = torch.device(device)
        self.n_slices_3d = n_slices_3d

        # Carga el modelo timm completo
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # elimina head de clasificación
        )
        self.vit.to(self.device)

        # d_model = dimensión del CLS token
        self.d_model = self.vit.embed_dim  # 192 para vit_tiny

        if freeze:
            self._freeze()

        self.eval()

    # ── Congelado ─────────────────────────────────────────────────────────

    def _freeze(self):
        for p in self.vit.parameters():
            p.requires_grad = False

    # ── Forward ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Entrada:
          x: (B, 3, 224, 224)       → imagen 2D preprocesada
          x: (B, 1, D, H, W)        → volumen 3D preprocesado

        Salida:
          z: (B, d_model)   CLS token
        """
        if x.dim() == 4:
            return self._forward_2d(x)
        elif x.dim() == 5:
            return self._forward_3d(x)
        else:
            raise ValueError(f"Input rank inesperado: {x.dim()}")

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) → (B, d_model)."""
        x = x.to(self.device)
        return self.vit(x)                  # timm num_classes=0 → CLS token

    def _forward_3d(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, 1, D, H, W) → (B, d_model)
        Slice-pooling: toma N slices centrales en el eje D,
        los convierte a RGB, pasa por el ViT, y promedia.
        """
        import torch.nn.functional as F

        x   = x.to(self.device)
        B, C, D, H, W = x.shape

        # Seleccionar slices centrales
        start  = max(0, D // 2 - self.n_slices_3d // 2)
        end    = min(D, start + self.n_slices_3d)
        slices = x[:, :, start:end, :, :]  # (B, 1, n, H, W)

        # Interpolate slices to 224x224 si hace falta
        n   = slices.shape[2]
        cls_list = []

        for i in range(n):
            sl = slices[:, :, i, :, :]                        # (B, 1, H, W)
            sl = sl.repeat(1, 3, 1, 1)                        # (B, 3, H, W)
            sl = F.interpolate(sl, size=(224, 224),
                               mode="bilinear", align_corners=False)
            cls_list.append(self.vit(sl))                      # (B, d_model)

        # Promediar sobre slices → (B, d_model)
        z = torch.stack(cls_list, dim=1).mean(dim=1)
        return z


# ── Factory function ──────────────────────────────────────────────────────────

def build_backbone(
    model_name:  str  = "vit_tiny_patch16_224",
    pretrained:  bool = True,
    device:      str  = "cpu",
    n_slices_3d: int  = 8,
) -> SharedBackbone:
    """Construye y devuelve el backbone listo para usar."""
    bb = SharedBackbone(
        model_name=model_name,
        pretrained=pretrained,
        freeze=True,
        device=device,
        n_slices_3d=n_slices_3d,
    )
    print(f"[Backbone] {model_name} | d_model={bb.d_model} | "
          f"device={bb.device} | parámetros congelados")
    return bb
