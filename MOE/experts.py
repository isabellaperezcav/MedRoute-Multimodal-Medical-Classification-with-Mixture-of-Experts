"""
experts.py — Carga de los 5 expertos médicos
Cada experto tiene arquitectura distinta. Este módulo reconstruye cada
arquitectura y carga sus pesos desde checkpoint.

Mapa de expertos
----------------
  0 → Experto 1 (NIH ChestX-ray14) → ConvNeXt-Tiny   | 6 clases  multilabel
  1 → Experto 2 (Osteoartritis)     → EfficientNet-B0  | 5 clases
  2 → Experto 3 (ISIC 2019)         → ConvNeXt-Small   | 8 clases
  3 → Experto 4 (LUNA16 3D)         → DenseNet 3D      | 2 clases binario
  4 → Experto 5 (Pancreas CT 3D)    → ResNet 3D        | 2 clases binario
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  Arquitecturas 3D propias (DenseNet3D y ResNet3D)
#  Los expertos 2D usan timm/torchvision; los 3D están definidos aquí porque
#  no hay una versión estándar de torchvision en 3D.
# ─────────────────────────────────────────────────────────────────────────────

class _ConvBnRelu3D(nn.Sequential):
    def __init__(self, in_c, out_c, **kw):
        super().__init__(
            nn.Conv3d(in_c, out_c, bias=False, **kw),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )


class _DenseLayer3D(nn.Module):
    def __init__(self, in_features, growth_rate=32, bn_size=4, drop_rate=0.0):
        super().__init__()
        inter = bn_size * growth_rate
        self.bn1 = nn.BatchNorm3d(in_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_features, inter, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(inter)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(inter, growth_rate, 3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        if self.drop_rate > 0:
            out = nn.functional.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], dim=1)


class _DenseBlock3D(nn.Module):
    def __init__(self, num_layers, in_features, growth_rate=32, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                _DenseLayer3D(in_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition3D(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.BatchNorm3d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_c, out_c, 1, bias=False),
            nn.AvgPool3d(2, stride=2),
        )


class DenseNet3D(nn.Module):
    """
    DenseNet-121 3D liviano para volúmenes (1, 64, 64, 64).
    Parámetros ajustados para coincidir con el entrenamiento de LUNA16.
    """
    def __init__(self, num_classes=2, growth_rate=16, block_config=(4, 4, 4, 4),
                 init_features=32, bn_size=4, drop_rate=0.0):
        super().__init__()
        # Stem
        self.features = nn.Sequential(
            nn.Conv3d(1, init_features, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),
        )
        num_features = init_features
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock3D(num_layers, num_features, growth_rate, bn_size, drop_rate)
            self.blocks.append(block)
            num_features += num_layers * growth_rate
            if i < len(block_config) - 1:
                trans = _Transition3D(num_features, num_features // 2)
                self.transitions.append(trans)
                num_features = num_features // 2
        self.norm_final = nn.BatchNorm3d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        x = torch.relu(self.norm_final(x))
        x = nn.functional.adaptive_avg_pool3d(x, 1).view(x.size(0), -1)
        return self.classifier(x)


class _ResBlock3D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_c)
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_c),
            )

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return torch.relu(out + identity)


class ResNet3D(nn.Module):
    """
    ResNet-18 3D liviano para volúmenes (1, 64, 64, 64).
    Arquitectura usada por Experto 5 (Pancreas CT).
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(_ResBlock3D(32,  64,  stride=1),
                                     _ResBlock3D(64,  64))
        self.layer2 = nn.Sequential(_ResBlock3D(64,  128, stride=2),
                                     _ResBlock3D(128, 128))
        self.layer3 = nn.Sequential(_ResBlock3D(128, 256, stride=2),
                                     _ResBlock3D(256, 256))
        self.pool  = nn.AdaptiveAvgPool3d(1)
        self.fc    = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Funciones de construcción y carga por experto
# ─────────────────────────────────────────────────────────────────────────────

def _load_checkpoint(model: nn.Module, ckpt_path: Path,
                     state_dict_key: Optional[str] = None) -> nn.Module:
    """
    Carga pesos desde un checkpoint PyTorch.
    Maneja formato completo (torch.save(model)) y solo state_dict.
    Intenta quitar prefijos "module." de DataParallel si hace falta.
    """
    raw = torch.load(str(ckpt_path), map_location="cpu")

    # Si el checkpoint ES el modelo completo
    if isinstance(raw, nn.Module):
        return raw

    # Si es un dict con clave específica
    if isinstance(raw, dict):
        if state_dict_key and state_dict_key in raw:
            sd = raw[state_dict_key]
        elif "model_state_dict" in raw:
            sd = raw["model_state_dict"]
        elif "state_dict" in raw:
            sd = raw["state_dict"]
        else:
            sd = raw   # asumir que todo el dict es el state_dict

        # Limpiar prefijo DataParallel
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

        # Cargar con strict=False para tolerar diferencias menores
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [warn] keys faltantes: {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"  [warn] keys inesperadas: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    else:
        raise ValueError(f"Formato de checkpoint no reconocido: {type(raw)}")

    return model


def _build_expert1(ckpt_path: Path) -> nn.Module:
    """Experto 1 — ChestX-ray14 → ConvNeXt-Tiny (6 clases multilabel)."""
    import timm
    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=6)
    print(f"  [Exp1] ConvNeXt-Tiny | 6 clases | {ckpt_path.name}")
    return _load_checkpoint(model, ckpt_path)


def _build_expert2(ckpt_path: Path) -> nn.Module:
    """Experto 2 — Osteoartritis → EfficientNet-B0 + cabeza personalizada (5 clases)."""
    import timm
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
    print(f"  [Exp2] EfficientNet-B0 | 5 clases | {ckpt_path.name}")
    return _load_checkpoint(model, ckpt_path)


def _build_expert3(ckpt_path: Path) -> nn.Module:
    """Experto 3 — ISIC 2019 → ConvNeXt-Small (8 clases)."""
    import timm
    model = timm.create_model("convnext_small", pretrained=False, num_classes=8)
    print(f"  [Exp3] ConvNeXt-Small | 8 clases | {ckpt_path.name}")
    return _load_checkpoint(model, ckpt_path)


def _build_expert4(ckpt_path: Path) -> nn.Module:
    """Experto 4 — LUNA16 → DenseNet 3D (2 clases)."""
    model = DenseNet3D(num_classes=2)
    print(f"  [Exp4] DenseNet3D | 2 clases | {ckpt_path.name}")
    return _load_checkpoint(model, ckpt_path)


def _build_expert5(ckpt_path: Path) -> nn.Module:
    """Experto 5 — Pancreas CT → ResNet 3D (2 clases)."""
    model = ResNet3D(num_classes=2)
    print(f"  [Exp5] ResNet3D | 2 clases | {ckpt_path.name}")
    return _load_checkpoint(model, ckpt_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Loader principal
# ─────────────────────────────────────────────────────────────────────────────

# Mapa: expert_id → (carpeta_relativa, nombre_ckpt, builder_fn)
_EXPERT_CONFIG = {
    0: ("expertos/experto1_Xray",     "ckpt_exp1v16_f2_best.pt",  _build_expert1),
    1: ("expertos/experto2_osteo",    "expert2_osteo_best.pth",   _build_expert2),
    2: ("expertos/experto3_isic",     "expert3_isic_best.pth",    _build_expert3),
    3: ("expertos/experto4_luna",     "LUNA-LIDCIDRI_best.pt",    _build_expert4),
    4: ("expertos/experto5_pancreas", "exp5_best.pth",            _build_expert5),
}

# Número de clases y tipo de tarea por experto
EXPERT_META = {
    0: {"name": "ChestX-ray14", "num_classes": 6,  "task": "multilabel"},
    1: {"name": "Osteoartritis","num_classes": 5,  "task": "multiclass"},
    2: {"name": "ISIC2019",     "num_classes": 8,  "task": "multiclass"},
    3: {"name": "LUNA16",       "num_classes": 2,  "task": "binary"},
    4: {"name": "Pancreas",     "num_classes": 2,  "task": "binary"},
}


def load_all_experts(
    base_dir: Union[str, Path] = ".",
    device:   str = "cpu",
) -> Dict[int, nn.Module]:
    """
    Carga los 5 expertos y los devuelve en un dict {expert_id: model}.
    Todos se ponen en modo eval() y se mueven al device indicado.

    Parámetros
    ----------
    base_dir : directorio raíz del proyecto (donde está la carpeta expertos/)
    device   : "cpu" o "cuda"
    """
    base   = Path(base_dir)
    dev    = torch.device(device)
    experts = {}

    print("[Experts] Cargando expertos...")
    for eid, (folder, ckpt_name, builder) in _EXPERT_CONFIG.items():
        ckpt_path = base / folder / ckpt_name
        if not ckpt_path.exists():
            # Buscar cualquier .pt/.pth en la carpeta si el nombre exacto falla
            folder_path = base / folder
            candidates  = list(folder_path.glob("*.pt")) + list(folder_path.glob("*.pth"))
            if candidates:
                ckpt_path = candidates[0]
                print(f"  [warn] Checkpoint exacto no encontrado, usando: {ckpt_path.name}")
            else:
                raise FileNotFoundError(
                    f"No se encontró checkpoint para experto {eid} en {folder_path}"
                )

        model = builder(ckpt_path)
        model.eval()
        model.to(dev)
        experts[eid] = model

    print(f"[Experts] {len(experts)} expertos cargados en {device}")
    return experts


def load_single_expert(
    expert_id: int,
    base_dir:  Union[str, Path] = ".",
    device:    str = "cpu",
) -> nn.Module:
    """Carga un único experto por ID (útil para pruebas)."""
    base = Path(base_dir)
    folder, ckpt_name, builder = _EXPERT_CONFIG[expert_id]
    ckpt_path = base / folder / ckpt_name
    model = builder(ckpt_path)
    model.eval()
    model.to(torch.device(device))
    return model
