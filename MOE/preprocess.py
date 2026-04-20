"""
preprocess.py — Preprocesador Adaptativo 2D/3D
Detecta automáticamente el tipo de entrada por shape del tensor.
Soporta PNG/JPG (2D) y NIfTI .nii/.nii.gz (3D).
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Union, Tuple

# ── Constantes ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
HU_MIN, HU_MAX = -1000.0, 400.0
SIZE_2D = 224
SIZE_3D = 64


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_2d(path: Union[str, Path]) -> torch.Tensor:
    """PNG/JPG → tensor (1, 3, H, W) float32 [0,1]."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]


def load_3d_nifti(path: Union[str, Path]) -> torch.Tensor:
    """NIfTI → tensor (1, 1, D, H, W) float32."""
    import nibabel as nib
    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    return torch.from_numpy(data).unsqueeze(0).unsqueeze(0)      # [1,1,D,H,W]


def load_3d_numpy(path: Union[str, Path]) -> torch.Tensor:
    """NumPy .npy → tensor float32, infiere 2D o 3D por ndim."""
    data = np.load(str(path)).astype(np.float32)
    if data.ndim == 3:
        return torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    elif data.ndim == 4:
        return torch.from_numpy(data).unsqueeze(0)               # [1,C,D,H,W]
    raise ValueError(f"NumPy shape inesperado: {data.shape}")


# ── Normalización ─────────────────────────────────────────────────────────────

def normalize_2d(t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32,
                        device=t.device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32,
                        device=t.device).view(1, 3, 1, 1)
    return (t - mean) / std


def normalize_3d(t: torch.Tensor) -> torch.Tensor:
    """
    Normaliza un volumen 3D a [0, 1].

    [FIX pipeline-3D] Detecta si el volumen YA viene normalizado en [0, 1]
    (caso de los .npy preprocesados offline como candidate_000123.npy de
    LUNA16_LIDCIDRI_transformation). Aplicarle HU clip otra vez colapsa
    todo a una constante ~0,714 y el ViT produce CLS genericos.

    - Si t en [0, 1] aprox: solo clamp defensivo, sin re-normalizar.
    - Si t en HU crudo: clamp a [-1000, 400] y escalar a [0, 1].
    """
    t_min = float(t.min())
    t_max = float(t.max())
    if t_min >= -0.1 and t_max <= 1.1:
        return t.clamp(0.0, 1.0)
    t = t.clamp(HU_MIN, HU_MAX)
    return (t - HU_MIN) / (HU_MAX - HU_MIN)


# ── Clase principal ───────────────────────────────────────────────────────────

class AdaptivePreprocessor:
    """
    Pipeline adaptativo:
      rank=4 → 2D → resize(224,224) + ImageNet norm
      rank=5 → 3D → HU clip + resize(64,64,64)

    Uso:
        prep  = AdaptivePreprocessor(device="cuda")
        x, mod = prep.from_file("scan.nii.gz")   # mod ∈ {"2D","3D"}
        x, mod = prep.from_tensor(raw_tensor)
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    # ── API pública ───────────────────────────────────────────────────────

    def from_file(self, path: Union[str, Path]) -> Tuple[torch.Tensor, str]:
        path   = Path(path)
        suffix = "".join(path.suffixes).lower()

        if suffix in (".nii", ".nii.gz"):
            t = load_3d_nifti(path).to(self.device)
            return self._process_3d(t)
        elif suffix == ".npy":
            t = load_3d_numpy(path).to(self.device)
            return self._process_3d(t) if t.dim() == 5 else self._process_2d(t)
        elif suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"):
            t = load_2d(path).to(self.device)
            return self._process_2d(t)
        else:
            raise ValueError(f"Extensión no soportada: {suffix}")

    def from_tensor(self, t: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """Detecta 2D/3D por rank del tensor (4 o 5)."""
        t = t.to(self.device)
        if t.dim() == 4:
            return self._process_2d(t)
        elif t.dim() == 5:
            return self._process_3d(t)
        raise ValueError(f"Rank inesperado: {t.dim()} — se esperaba 4 (2D) ó 5 (3D)")

    # ── Pipelines internos ────────────────────────────────────────────────

    def _process_2d(self, t: torch.Tensor) -> Tuple[torch.Tensor, str]:
        if t.shape[1] == 1:
            t = t.repeat(1, 3, 1, 1)               # escala de grises → RGB
        t = F.interpolate(t, size=(SIZE_2D, SIZE_2D),
                          mode="bilinear", align_corners=False)
        t = normalize_2d(t)
        return t, "2D"

    def _process_3d(self, t: torch.Tensor) -> Tuple[torch.Tensor, str]:
        t = normalize_3d(t)
        t = F.interpolate(t, size=(SIZE_3D, SIZE_3D, SIZE_3D),
                          mode="trilinear", align_corners=False)
        return t, "3D"
