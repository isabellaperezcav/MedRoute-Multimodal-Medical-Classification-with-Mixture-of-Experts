"""
dashboard/app.py — Dashboard MoE Médico (Integrado con pipeline real)
Proyecto Final — Incorporar Elementos de IA, Unidad II (Bloque Visión)

Pipeline real:
  archivo/tensor → AdaptivePreprocessor → SharedBackbone (CLS)
                → KNNRouter (FAISS) → Expert_k → predicción
"""

# ─── Imports estándar ────────────────────────────────────────────────────────
import sys
import io
import time
import json
import warnings
import logging
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

warnings.filterwarnings("ignore")
logging.getLogger("timm").setLevel(logging.ERROR)   # silenciar logs de timm

# ─── Integración con el módulo MOE ───────────────────────────────────────────
#   El dashboard vive en dashboard/app.py y el módulo en MOE/
#   Añadimos el directorio raíz del proyecto al path para importar correctamente.
ROOT_DIR = Path(__file__).resolve().parent.parent   # sube un nivel desde dashboard/
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MOE_AVAILABLE = False
try:
    from MOE.preprocess  import AdaptivePreprocessor
    from MOE.backbone    import build_backbone
    from MOE.router_knn  import build_router
    from MOE.experts     import load_all_experts, EXPERT_META
    from MOE.moe_model   import MoEModel
    import torch
    MOE_AVAILABLE = True
except ImportError as _e:
    # El dashboard degradará a modo demo cuando el MoE no esté disponible.
    # Esto permite que la UI funcione incluso sin checkpoints instalados.
    _MOE_IMPORT_ERROR = str(_e)

# ─── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="MoE Medical Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 20px 30px; border-radius: 12px;
        margin-bottom: 20px; color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8em; }
    .main-header p  { color: #cce3f5; margin: 4px 0 0 0; font-size: 0.95em; }

    .metric-card {
        background: #f0f6ff; border: 1px solid #c0d8f0;
        border-radius: 10px; padding: 16px;
        text-align: center; margin-bottom: 10px;
    }
    .metric-card h3 { margin: 0; color: #1e3a5f; font-size: 2em; }
    .metric-card p  { margin: 0; color: #555; font-size: 0.85em; }

    .expert-card {
        background: linear-gradient(135deg, #e8f4fd, #d0e9f9);
        border-left: 5px solid #2d6a9f;
        border-radius: 8px; padding: 14px; margin: 8px 0;
    }
    .expert-card h4 { color: #1e3a5f; margin: 0 0 6px 0; }
    .expert-card p  { color: #333; margin: 2px 0; font-size: 0.88em; }

    .ood-alert {
        background: #fff3cd; border: 2px solid #ffc107;
        border-radius: 10px; padding: 14px; margin: 8px 0;
    }
    .ood-ok {
        background: #d4edda; border: 2px solid #28a745;
        border-radius: 10px; padding: 14px; margin: 8px 0;
    }
    .section-title {
        font-size: 1.05em; font-weight: 700; color: #1e3a5f;
        border-bottom: 2px solid #2d6a9f;
        padding-bottom: 4px; margin: 16px 0 10px 0;
    }
    .badge {
        display: inline-block; padding: 3px 10px;
        border-radius: 20px; font-size: 0.78em; font-weight: 600;
    }
    .badge-green  { background:#d4edda; color:#155724; }
    .badge-blue   { background:#cce5ff; color:#004085; }
    .badge-orange { background:#fff3cd; color:#856404; }
    .badge-red    { background:#f8d7da; color:#721c24; }
    .mode-banner {
        background: #fff3cd; border: 1px solid #ffc107;
        border-radius: 8px; padding: 8px 14px; margin-bottom: 10px;
        font-size: 0.87em; color: #856404;
    }
    .mode-banner-real {
        background: #d4edda; border: 1px solid #28a745;
        border-radius: 8px; padding: 8px 14px; margin-bottom: 10px;
        font-size: 0.87em; color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# ─── Metadatos de expertos (fuente de verdad visual del dashboard) ────────────
# Nota: EXPERT_META en experts.py contiene task/num_classes.
# Aquí añadimos información adicional para la UI.
# Fuente de verdad: moe/experts/wrappers.py::EXPERT_SPECS.
# El router fue entrenado con embeddings/-.py EXPERT_MAP = {chest:0, isic:1, osteo:2, luna:3, panc:4}.
# Los nombres de archivo "expert2_osteo" y "expert3_isic" son heredados del PDF y estan invertidos
# respecto al label real del router (ver nota "naming local invertido" en wrappers.py:92).
EXPERTS_UI = {
    0: {"name": "Exp1 — ChestX-ray14",  "arch": "ConvNeXt-V2 Base 384", "dataset": "NIH ChestX-ray14",   "task": "Multilabel 6 patologías"},
    1: {"name": "Exp2 — ISIC 2019",     "arch": "EfficientNet-B3",      "dataset": "ISIC 2019",           "task": "Multiclase 8 clases dermato."},
    2: {"name": "Exp3 — Osteoartritis", "arch": "EfficientNet-B0",      "dataset": "OA Knee X-ray",       "task": "5 grados KL"},
    3: {"name": "Exp4 — LUNA16 CT",     "arch": "DenseNet-3D",          "dataset": "LUNA16 / LIDC-IDRI",  "task": "Nódulo pulmonar 2 clases"},
    4: {"name": "Exp5 — Páncreas CT",   "arch": "R3D-18",               "dataset": "Pancreatic Cancer CT","task": "Tumor 2 clases"},
}

# Las listas de etiquetas tienen que coincidir con EXPERT_SPECS[label].class_names:
#   ISIC 2019: 8 clases MEL/NV/BCC/AK/BKL/DF/VASC/SCC (ojo: "AK", no "AKIEC", segun wrappers.py:87).
#   Osteoartritis: 5 clases Normal/Dudoso/Leve/Moderado/Severo (wrappers.py:103).
LABELS_UI = {
    0: ["Infiltration", "Effusion", "Atelectasis", "Nodule", "Mass", "Pneumothorax"],
    1: ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"],
    2: ["KL-0 Normal", "KL-1 Dudoso", "KL-2 Leve", "KL-3 Moderado", "KL-4 Severo"],
    3: ["Sin nódulo", "Con nódulo"],
    4: ["Normal", "Con tumor"],
}

OOD_THRESHOLD  = 1.35
ABLATION_JSON  = ROOT_DIR / "ablation" / "outputs" / "ablation" / "ablation_results_v2.json"
EMBEDDINGS_DIR = ROOT_DIR / "embedings" / "Output embeddings"

# ─── Estado de sesión ─────────────────────────────────────────────────────────
# NOTA: f_i_history se inicializa a ceros (no a [0.20]*5) para que
# compute_load_ratio distinga inequívocamente "sin inferencias" (suma=0)
# de "5 expertos activos con carga uniforme" (suma=1, cada entrada 0.20).
for key, default in [
    ("f_i_history",  np.zeros(5, dtype=np.float64)),
    ("n_inferences", 0),
    ("last_result",  None),
    ("batch_results", []),    # lista de dicts por imagen procesada en batch
    ("batch_files_sig", None), # firma del batch actual para detectar cambios
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE MODELO (cached — se ejecuta UNA vez por sesión)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Cargando sistema MoE (primera vez puede tardar ~30 s)...")
def load_moe_model():
    """
    Carga el MoEModel completo: backbone + router FAISS + 5 expertos.
    Se cachea en sesión para no recargar en cada interacción.
    Devuelve (model, error_str).  error_str es None si todo OK.
    """
    if not MOE_AVAILABLE:
        return None, f"MOE no importable: {_MOE_IMPORT_ERROR}"
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = MoEModel(
            base_dir       = str(ROOT_DIR),
            device         = device,
            use_fp16       = False,          # FP16 solo si GPU confirmada
            knn_k          = 5,
            embeddings_dir = str(EMBEDDINGS_DIR),
        )
        model.eval()
        return model, None
    except Exception as exc:
        return None, str(exc)


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCIA REAL (usando pipeline MOE)
# ══════════════════════════════════════════════════════════════════════════════

def real_inference(img_pil: Image.Image, is_nifti_input: bool, nifti_volume=None) -> dict:
    """
    Ejecuta el pipeline completo del MoE sobre la imagen/volumen recibido.

    Flujo:
        PIL/volume → preprocesador → backbone CLS → router kNN → experto → logits

    Devuelve dict compatible con el formato del dashboard (mismo esquema
    que mock_inference para no cambiar el layout de resultados).
    """
    model, err = load_moe_model()

    if model is None:
        # Fallback a demo si el modelo no cargó
        return _demo_inference(img_pil, err)

    t0 = time.perf_counter()

    try:
        import torch
        import torch.nn.functional as F

        if is_nifti_input and nifti_volume is not None:
            # ── Volumen 3D ────────────────────────────────────────────────
            vol_t = torch.from_numpy(
                nifti_volume.astype(np.float32)
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            output, expert_id, router_info = model.forward(vol_t)
        else:
            # ── Imagen 2D ─────────────────────────────────────────────────
            arr = np.array(img_pil.convert("RGB")).astype(np.float32) / 255.0
            t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
            output, expert_id, router_info = model.forward(t)

        t1 = time.perf_counter()

        probs = output[0].detach().cpu().float().numpy()
        labels = LABELS_UI.get(expert_id, [f"class_{i}" for i in range(len(probs))])
        pred_idx   = int(np.argmax(probs))
        pred_label = labels[pred_idx] if pred_idx < len(labels) else f"class_{pred_idx}"
        confidence = float(probs[pred_idx])

        # Gating scores: cosine_scores de los k vecinos → normalizar a 5 expertos
        gating = _build_gating_from_router(router_info)

        # Attention heatmap: generar desde el embedding CLS del backbone
        attn_map = _build_attention_map(img_pil, model, is_nifti_input, nifti_volume)

        entropy = float(-np.sum(gating * np.log(gating + 1e-8)))

        return {
            "expert_idx":   expert_id,
            "gating":       gating,
            "pred_label":   pred_label,
            "pred_idx":     pred_idx,
            "confidence":   confidence,
            "probs":        probs,
            "labels":       labels,
            "attn_map":     attn_map,
            "entropy":      entropy,
            "latency_ms":   (t1 - t0) * 1000,
            "router_info":  router_info,
            "mode":         "real",
        }

    except Exception as exc:
        # Si la inferencia falla (ej: checkpoint con llave distinta), usar demo
        st.warning(f"Inferencia real falló: `{exc}`. Mostrando demo.")
        return _demo_inference(img_pil, str(exc))


def _build_gating_from_router(router_info: dict) -> np.ndarray:
    """
    Convierte la info del router kNN (votos por experto) en un vector
    de 5 scores pseudo-gating normalizados a [0,1] que suma 1.

    Usa los cosine_scores de los k vecinos ponderados por experto.
    """
    vote_counts  = router_info.get("vote_counts", {})
    cosine_scores = router_info.get("cosine_scores", [])
    neighbor_labels = router_info.get("neighbor_labels", [])

    gating = np.zeros(5, dtype=np.float32)

    if cosine_scores and neighbor_labels:
        # Sumar cosine score de cada vecino al experto que representa
        for lbl, sc in zip(neighbor_labels, cosine_scores):
            eid = int(lbl)
            if 0 <= eid < 5:
                gating[eid] += max(float(sc), 0.0)
    elif vote_counts:
        for eid, cnt in vote_counts.items():
            if 0 <= int(eid) < 5:
                gating[int(eid)] = float(cnt)

    total = gating.sum()
    if total > 0:
        gating /= total
    else:
        # Si algo falla, poner score 1.0 al experto seleccionado
        gating[router_info.get("expert_id", 0)] = 1.0

    return gating


def _build_attention_map(img_pil, model, is_nifti, volume=None) -> np.ndarray:
    """
    Genera un heatmap de atención aproximado usando la norma del embedding
    por patch del ViT (ViT Attention-based saliency liviana).

    Si el modelo no soporta hooks, devuelve un mapa gaussiano centrado
    en la región de mayor varianza de la imagen.
    """
    try:
        import torch
        import torch.nn.functional as F

        # Método: hook en el último bloque de atención del ViT
        attn_weights = []

        def _hook(module, input, output):
            # output puede ser (attn_output, attn_weights) o solo attn_output
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights.append(output[1].detach().cpu())

        # Registrar hook en el último bloque ViT
        hooks = []
        try:
            last_block = list(model.backbone.vit.blocks)[-1]
            h = last_block.attn.register_forward_hook(_hook)
            hooks.append(h)
        except Exception:
            pass

        # Forward pass liviano solo para el backbone
        if is_nifti and volume is not None:
            vol_t = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            # Usar slice central para el heatmap
            D = vol_t.shape[2]
            sl = vol_t[:, :, D // 2, :, :]  # (1, 1, H, W)
            sl_rgb = sl.repeat(1, 3, 1, 1)
            sl_rgb = F.interpolate(sl_rgb, size=(224, 224), mode="bilinear", align_corners=False)
            img_for_attn = sl_rgb
        else:
            arr = np.array(img_pil.convert("RGB")).astype(np.float32) / 255.0
            t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            t   = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
            img_for_attn = t

        with torch.no_grad():
            model.backbone.vit(img_for_attn.to(model.device))

        for h in hooks:
            h.remove()

        if attn_weights:
            # attn_weights: (B, heads, N+1, N+1) — ignorar token CLS (idx 0)
            aw   = attn_weights[-1][0]           # (heads, N+1, N+1)
            # CLS → patches: fila 0, columnas 1:
            cls_attn = aw[:, 0, 1:].mean(0)     # (N,) promedio sobre heads
            cls_attn = cls_attn.numpy()
            side = int(np.sqrt(len(cls_attn)))   # 14 para patch_size=16, img=224
            attn_map = cls_attn.reshape(side, side)
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            # Upsample a 224×224
            attn_t   = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0)
            attn_224 = F.interpolate(attn_t, size=(224, 224),
                                     mode="bilinear", align_corners=False)
            return attn_224.squeeze().numpy()

    except Exception:
        pass

    # Fallback: mapa gaussiano centrado en zona de mayor varianza
    return _variance_heatmap(img_pil)


def _variance_heatmap(img_pil: Image.Image) -> np.ndarray:
    """Heatmap de saliencia basado en varianza local (sin necesidad de pesos)."""
    arr = np.array(img_pil.convert("L").resize((224, 224))).astype(np.float32)
    from scipy.ndimage import uniform_filter
    mean   = uniform_filter(arr,    size=16)
    mean_sq = uniform_filter(arr**2, size=16)
    var    = mean_sq - mean**2
    var    = np.clip(var, 0, None)
    if var.max() > 0:
        var /= var.max()
    return var


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO INFERENCE (fallback si el modelo no carga)
# ══════════════════════════════════════════════════════════════════════════════

def _demo_inference(img_pil: Image.Image, error_msg: str = "") -> dict:
    """
    Inferencia de demostración (valores simulados).
    Se activa cuando el modelo real no está disponible (sin checkpoints).
    """
    np.random.seed(int(time.time() * 1000) % 2**31)
    t0 = time.perf_counter()

    raw        = np.random.dirichlet(np.ones(5) * 3)
    expert_idx = int(np.argmax(raw))
    labels     = LABELS_UI[expert_idx]
    n_labels   = len(labels)
    logits     = np.random.randn(n_labels)
    probs      = np.exp(logits) / np.exp(logits).sum()
    pred_idx   = int(np.argmax(probs))
    pred_label = labels[pred_idx]
    confidence = float(np.max(probs))

    # Attn heatmap gaussiana simulada
    h, w = 224, 224
    cx, cy = np.random.randint(50, 174), np.random.randint(50, 174)
    Y, X   = np.mgrid[0:h, 0:w]
    attn   = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * 40**2))
    attn  += np.random.rand(h, w) * 0.15
    attn   = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    entropy = float(-np.sum(raw * np.log(raw + 1e-8)))
    t1      = time.perf_counter()

    return {
        "expert_idx":   expert_idx,
        "gating":       raw,
        "pred_label":   pred_label,
        "pred_idx":     pred_idx,
        "confidence":   confidence,
        "probs":        probs,
        "labels":       labels,
        "attn_map":     attn,
        "entropy":      entropy,
        "latency_ms":   (t1 - t0) * 1000,
        "router_info":  {},
        "mode":         "demo",
        "demo_reason":  error_msg,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCIONES AUXILIARES UI
# ══════════════════════════════════════════════════════════════════════════════

def detect_ood(result: dict, entropy_thr: float, cos_thr: float) -> dict:
    """
    Decide si una imagen es OOD combinando dos señales complementarias:

    1. Entropía del gating (votos kNN dispersos) → indeterminación del router.
    2. Similitud coseno con el vecino más cercano (distancia mínima al training
       set en el espacio CLS) → qué tan lejos cae la imagen de cualquier
       dominio conocido.

    Ambas señales son necesarias porque por separado fallan:
    - Entropía sola: una foto de perro puede concentrar sus 5 vecinos en un
      único experto por azar y dar entropía baja pese a estar lejos de todo.
    - Coseno solo: una imagen ambigua entre dos modalidades médicas (p. ej.
      una radiografía con fuerte artefacto) puede tener coseno alto pero
      entropía alta por confusión entre expertos legítimos.

    La combinación OR captura ambos regímenes.

    Retorna dict con is_ood, reasons, max_cos, entropy.
    """
    entropy = float(result.get("entropy", 0.0))
    router_info   = result.get("router_info", {}) or {}
    cosine_scores = router_info.get("cosine_scores", []) or []
    max_cos = float(max(cosine_scores)) if cosine_scores else 1.0

    reasons = []
    if entropy > entropy_thr:
        reasons.append(f"entropía {entropy:.3f} > {entropy_thr:.2f}")
    if max_cos < cos_thr:
        reasons.append(f"coseno máx {max_cos:.3f} < {cos_thr:.2f}")

    return {
        "is_ood":  len(reasons) > 0,
        "reasons":" y ".join(reasons) if reasons else "",
        "max_cos": max_cos,
        "entropy": entropy,
    }


def fmt_es(value: float, decimals: int = 3) -> str:
    """Formatea un float con coma decimal estilo español."""
    return f"{value:.{decimals}f}".replace(".",",")


def update_load_balance(expert_idx):
    expert_idx = int(expert_idx)
    if not (0 <= expert_idx < 5):
        return
    n         = st.session_state.n_inferences
    current   = np.asarray(st.session_state.f_i_history, dtype=np.float64)
    counts    = current * n
    counts[expert_idx] += 1
    total = counts.sum()
    if total <= 0:
        return
    st.session_state.f_i_history  = counts / total
    st.session_state.n_inferences = n + 1


def compute_load_ratio():
    """
    Cociente max/min de f_i, computado SOLO sobre los expertos que han
    recibido al menos una inferencia.

    - Sin inferencias: (None, 0).
    - Un solo experto activo: ratio = 1.0 (balance trivial).
    - Varios activos: max(activos) / min(activos), sin epsilon artificial.
    - La penalizacion (panel y sidebar) solo se dispara cuando los 5
      expertos estan activos, porque el paper define el cociente sobre
      los 5 dominios. Mientras falten, se muestra ratio entre activos
      con etiqueta explicita de cobertura.

    Retorna (ratio: float | None, n_active: int en [0, 5]).
    """
    fi = st.session_state.f_i_history
    n  = st.session_state.n_inferences
    if n == 0:
        return None, 0
    active = fi[fi > 0]
    if len(active) == 0:
        return None, 0
    ratio = float(active.max() / active.min())
    return ratio, int(len(active))


def overlay_heatmap(img_pil: Image.Image, attn: np.ndarray) -> np.ndarray:
    img_rgb  = np.array(img_pil.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
    colormap = cm.get_cmap("jet")(attn)[..., :3]
    blended  = 0.55 * img_rgb + 0.45 * colormap
    return np.clip(blended, 0, 1)


def is_nifti_bytes(file_bytes: bytes) -> bool:
    return file_bytes[:4] in [b'\x5c\x01\x00\x00', b'\x00\x00\x01\x5c']


def load_nifti_bytes(file_bytes: bytes, fname: str = ""):
    """
    Carga un NIfTI desde bytes en memoria. Devuelve (volume_np, shape_str).
    Acepta tanto .nii como .nii.gz — el sufijo del tempfile se elige en función
    del nombre original para que nibabel detecte la compresión correctamente.
    """
    try:
        import nibabel as nib
        import tempfile, os
        # Elegir sufijo correcto: nibabel necesita .nii.gz para gzip
        suffix = ".nii.gz" if fname.lower().endswith(".gz") else ".nii"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        img    = nib.load(tmp_path)
        volume = img.get_fdata(dtype=np.float32)
        os.unlink(tmp_path)
        return volume, str(volume.shape)
    except Exception as exc:
        st.warning(f"nibabel no disponible o archivo inválido: {exc}. Usando volumen simulado.")
        vol = np.random.randint(40, 180, (64, 64, 64), dtype=np.uint8).astype(np.float32)
        return vol, "(64, 64, 64) [simulado]"


# ══════════════════════════════════════════════════════════════════════════════
#  NUEVA FUNCIÓN — load_medical_input
#  Wrapper de entrada que detecta tipo de archivo y lo normaliza antes de
#  pasarlo al pipeline MoE. NO modifica preprocess.py ni ninguna función
#  existente. Solo prepara la "materia prima" que ya recibía real_inference().
# ══════════════════════════════════════════════════════════════════════════════

def load_medical_input(uploaded_file) -> dict:
    """
    Carga y normaliza cualquier input médico soportado:
        - PNG / JPG / JPEG  → imagen 2D
        - .npy              → volumen 3D (o imagen 2D si ndim <= 3 con C≤4)
        - .nii / .nii.gz    → volumen NIfTI 3D

    Retorna un dict estandarizado con TODOS los campos que necesita el
    bloque de UI y real_inference(). El código existente sigue funcionando
    sin cambios porque recibe exactamente los mismos tipos que antes.

    Campos devueltos
    ----------------
    data          : PIL.Image  (2D) o np.ndarray float32 (3D, shape D×H×W)
    modality      : "2D" | "3D"
    orig_shape    : tuple  — shape original antes de cualquier resize
    is_volumetric : bool   — True para .npy 3D y .nii
    file_type     : str    — "png/jpg" | "npy" | "nifti"
    preview_pil   : PIL.Image — slice central para visualizar en la UI
    volume        : np.ndarray | None — array 3D float32, None si 2D
    n_slices      : int    — número de slices en eje Z (1 si 2D)
    error         : str | None — mensaje de error si algo falló
    """
    result = {
        "data":          None,
        "modality":      "2D",
        "orig_shape":    (),
        "is_volumetric": False,
        "file_type":     "unknown",
        "preview_pil":   None,
        "volume":        None,
        "n_slices":      1,
        "error":         None,
    }

    try:
        file_bytes = uploaded_file.read()
        fname      = uploaded_file.name.lower()

        # ── 1. NIfTI (.nii / .nii.gz) ─────────────────────────────────────
        if fname.endswith((".nii", ".nii.gz")) or is_nifti_bytes(file_bytes):
            volume, _ = load_nifti_bytes(file_bytes, fname=fname)   # fname → sufijo correcto para .nii.gz
            volume    = volume.astype(np.float32)

            result["file_type"]     = "nifti"
            result["modality"]      = "3D"
            result["is_volumetric"] = True
            result["orig_shape"]    = volume.shape
            result["volume"]        = volume
            result["n_slices"]      = volume.shape[2] if volume.ndim >= 3 else 1

            # Preview: slice central normalizado a [0,255]
            mid      = result["n_slices"] // 2
            sl       = volume[:, :, mid] if volume.ndim >= 3 else volume
            sl_u8    = _normalize_slice_to_uint8(sl)
            result["preview_pil"] = Image.fromarray(sl_u8).convert("RGB")
            result["data"]        = volume

        # ── 2. NumPy .npy ──────────────────────────────────────────────────
        elif fname.endswith(".npy"):
            arr = np.load(io.BytesIO(file_bytes)).astype(np.float32)

            # Detectar si es 2D o 3D por shape
            #   ndim == 2             → (H, W)            → 2D grayscale
            #   ndim == 3, C <= 4     → (H, W, C)         → 2D imagen RGB/RGBA/L
            #   ndim == 3, C >> 4     → (D, H, W)         → 3D volumen
            #   ndim == 4             → (D, H, W, C) o (C, D, H, W) → 3D volumen
            is_3d = _npy_is_3d(arr)

            if is_3d:
                # Asegurar orden (D, H, W) — si viene (C, D, H, W) con C pequeño, quitamos C
                volume = _npy_to_dhw(arr)

                result["file_type"]     = "npy"
                result["modality"]      = "3D"
                result["is_volumetric"] = True
                result["orig_shape"]    = volume.shape
                result["volume"]        = volume
                result["n_slices"]      = volume.shape[2] if volume.ndim >= 3 else 1

                mid   = result["n_slices"] // 2
                sl    = volume[:, :, mid] if volume.ndim >= 3 else volume
                sl_u8 = _normalize_slice_to_uint8(sl)
                result["preview_pil"] = Image.fromarray(sl_u8).convert("RGB")
                result["data"]        = volume

            else:
                # 2D numpy → convertir a PIL Image
                if arr.ndim == 2:
                    img_u8 = _normalize_slice_to_uint8(arr)
                    pil    = Image.fromarray(img_u8).convert("RGB")
                elif arr.ndim == 3:
                    # (H, W, C) con C ≤ 4
                    if arr.max() <= 1.0:
                        arr = (arr * 255).clip(0, 255)
                    pil = Image.fromarray(arr.astype(np.uint8))
                    if pil.mode not in ("RGB", "L"):
                        pil = pil.convert("RGB")
                else:
                    raise ValueError(f"Array .npy con shape inesperado: {arr.shape}")

                result["file_type"]   = "npy"
                result["modality"]    = "2D"
                result["orig_shape"]  = arr.shape
                result["preview_pil"] = pil
                result["data"]        = pil
                result["n_slices"]    = 1

        # ── 3. Imagen 2D estándar (PNG / JPG / JPEG / BMP / TIFF) ─────────
        else:
            pil = Image.open(io.BytesIO(file_bytes))
            result["file_type"]   = "png/jpg"
            result["modality"]    = "2D"
            result["orig_shape"]  = (*pil.size,)      # (W, H)
            result["preview_pil"] = pil
            result["data"]        = pil
            result["n_slices"]    = 1

    except Exception as exc:
        result["error"] = str(exc)
        # Devolver imagen negra como fallback para no romper la UI
        blank = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        result["preview_pil"] = blank
        result["data"]        = blank
        result["modality"]    = "error"
        result["orig_shape"]  = (224, 224)

    return result


# ── Helpers internos de load_medical_input ─────────────────────────────────────
#   Funciones privadas (prefijo _lmi_) — NO modifican ninguna función existente.

def _normalize_slice_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normaliza un array 2D float a uint8 [0,255] para visualización."""
    a = arr.astype(np.float32)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-8:
        return np.zeros_like(a, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255).astype(np.uint8)


def _npy_is_3d(arr: np.ndarray) -> bool:
    """
    Heurística para decidir si un array .npy es un volumen 3D médico.
    Regla: si ndim >= 3 y la dimensión más pequeña de las primeras dos
    axes es > 4, lo tratamos como volumen.
    """
    if arr.ndim == 4:
        return True
    if arr.ndim == 3:
        # (H, W, C) con C en {1,2,3,4} → 2D imagen
        if arr.shape[2] <= 4:
            return False
        # (D, H, W) con todas las dims grandes → 3D
        return True
    return False   # ndim <= 2 siempre es 2D


def _npy_to_dhw(arr: np.ndarray) -> np.ndarray:
    """
    Convierte array 3D/4D a forma canónica (D, H, W) float32.
    Maneja:  (D,H,W), (H,W,D), (C,D,H,W) con C pequeño, (D,H,W,C) con C pequeño.
    """
    if arr.ndim == 3:
        # Ya es (D, H, W) — tomamos tal cual
        return arr
    if arr.ndim == 4:
        # Si primera dimensión es pequeña (canal) → (C, D, H, W) → tomar canal 0
        if arr.shape[0] <= 4:
            return arr[0]
        # Si última dimensión es pequeña → (D, H, W, C) → tomar canal 0
        if arr.shape[3] <= 4:
            return arr[..., 0]
        # Asumir (D, H, W, extra) y colapsar
        return arr.mean(axis=-1)
    return arr


def _get_volume_slice(volume: np.ndarray, axis: int, idx: int) -> np.ndarray:
    """
    Extrae un slice 2D de un volumen (D, H, W) dado eje e índice.
    axis: 0=axial(Z), 1=coronal(Y), 2=sagital(X)
    """
    if axis == 0:
        sl = volume[idx, :, :]
    elif axis == 1:
        sl = volume[:, idx, :]
    else:
        sl = volume[:, :, idx]
    return sl


def load_ablation_data() -> dict:
    """
    Carga resultados del ablation study desde JSON.
    Fallback a datos hardcoded si el archivo no existe.
    """
    if ABLATION_JSON.exists():
        try:
            with open(ABLATION_JSON, "r") as f:
                raw = json.load(f)
            # Normalizar formato — el JSON puede tener distintas estructuras
            # Intentar extraer la tabla comparativa
            if isinstance(raw, list):
                # Lista de dicts con campos "router", "routing_accuracy", etc.
                routers, tipos, accs, lats, vrams, grads = [], [], [], [], [], []
                for entry in raw:
                    routers.append(entry.get("router", entry.get("name", "?")))
                    tipos.append(entry.get("type", entry.get("tipo", "?")))
                    accs.append(float(entry.get("routing_accuracy",
                                                entry.get("Routing Acc.", 0.0))))
                    lats.append(float(entry.get("latency_ms",
                                                entry.get("Latencia (ms)", 0.0))))
                    vrams.append(int(entry.get("vram_mb",
                                               entry.get("VRAM (MB)", 0))))
                    grads.append(entry.get("gradient", entry.get("Gradiente", "No")))
                return {
                    "Router":        routers,
                    "Tipo":          tipos,
                    "Routing Acc.":  accs,
                    "Latencia (ms)": lats,
                    "VRAM (MB)":     vrams,
                    "Gradiente":     grads,
                    "_source":       "json",
                }
            elif isinstance(raw, dict):
                raw["_source"] = "json"
                return raw
        except Exception as exc:
            st.caption(f"_No se pudo leer ablation JSON: {exc}_")

    # Fallback con los datos del ablation study real del proyecto
    return {
        "Router":        ["ViT + k-NN (FAISS)", "ViT + Linear (DL)", "ViT + GMM", "ViT + Naive Bayes"],
        "Tipo":          ["No paramétrico", "Paramétrico (grad.)", "Paramétrico (EM)", "Paramétrico (MLE)"],
        "Routing Acc.":  [1.0,  0.89, 0.83, 0.77],
        "Latencia (ms)": [4.8,  8.2,  3.1,  0.9],
        "VRAM (MB)":     [85,   1500, 50,   5],
        "Gradiente":     ["No", "Sí", "No", "No"],
        "_source":       "hardcoded",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>MoE Medical Imaging Dashboard</h1>
  <p>Clasificación Médica con Mixture of Experts + Ablation Study del Router &nbsp;|&nbsp;
     Incorporar Elementos de IA — Unidad II (Bloque Visión)</p>
</div>
""", unsafe_allow_html=True)

# Banner de modo (real vs demo)
if MOE_AVAILABLE:
    _model_check, _err = load_moe_model()
    if _model_check is not None:
        st.markdown('<div class="mode-banner-real"><strong>Modo REAL</strong> — '
                    'Pipeline MoE conectado (backbone ViT + router kNN FAISS + expertos)</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="mode-banner"><strong>Modo DEMO</strong> — '
                    f'MoE importado pero modelo no cargó: <code>{_err[:120]}</code></div>',
                    unsafe_allow_html=True)
else:
    st.markdown('<div class="mode-banner"><strong>Modo DEMO</strong> — '
                'Módulo MOE no encontrado. Verifica <code>sys.path</code> y checkpoints.</div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Configuración")

    router_choice = st.selectbox(
        "Router activo",
        ["ViT + k-NN (FAISS) ← ganador", "ViT + Linear (DL)",
         "ViT + GMM", "ViT + Naive Bayes"],
        index=0,
        help="El router k-NN FAISS es el ganador del ablation (Routing Acc = 1.0)."
    )

    show_ablation = st.checkbox("Mostrar panel Ablation Study", value=True)
    show_ood      = st.checkbox("Mostrar panel OOD Detection",  value=True)
    show_balance  = st.checkbox("Mostrar panel Load Balance",   value=True)

    st.divider()
    st.markdown("**Umbrales**")
    ood_threshold = st.slider(
        "Umbral entropía OOD", 0.5, 3.0, float(OOD_THRESHOLD), 0.05,
        help="Entropía del gating kNN. Alta entropía = votos dispersos entre expertos.",
    )
    ood_cos_threshold = st.slider(
        "Umbral coseno mínimo (OOD)", 0.10, 0.95, 0.50, 0.01,
        help=("Si la similitud coseno con el vecino más cercano cae por debajo "
              "de este umbral, la imagen se marca como OOD y no se delega al experto. "
              "Imágenes médicas reales: 0,6–0,98. Imágenes no médicas (p. ej. perros): 0,1–0,4."),
    )
    block_ood_inference = st.checkbox(
        "Bloquear predicción del experto si OOD",
        value=True,
        help="Si se activa y la imagen se detecta como OOD, el experto no recibe la imagen "
             "y tampoco se contabiliza en el balance de carga.",
    )
    balance_limit = st.slider("Límite cociente max/min f_i", 1.0, 3.0, 1.30, 0.05,
                               help="Penalización al superar este cociente.")

    st.divider()
    if st.button("Reiniciar contadores de carga"):
        st.session_state.f_i_history   = np.zeros(5, dtype=np.float64)
        st.session_state.n_inferences  = 0
        st.session_state.last_result   = None
        st.session_state.batch_results = []
        st.success("Contadores reiniciados")

    st.divider()
    st.caption(f"Total inferencias: **{st.session_state.n_inferences}**")
    ratio, n_active = compute_load_ratio()
    if ratio is None:
        st.caption(f"Cociente max/min f_i: **—** (sin inferencias, límite {fmt_es(balance_limit, 2)})")
    elif n_active < 5:
        st.caption(f"Cociente max/min f_i ({n_active}/5 activos): **{fmt_es(ratio, 3)}** (límite {fmt_es(balance_limit, 2)})")
    else:
        st.caption(f"Cociente max/min f_i: **{fmt_es(ratio, 3)}** (límite {fmt_es(balance_limit, 2)})")

    with st.expander("Debug Load Balance", expanded=False):
        fi_dbg = np.asarray(st.session_state.f_i_history, dtype=np.float64)
        st.code(
            "n_inferences = {n}\n"
            "f_i          = [{fi}]\n"
            "sum(f_i)     = {s:.6f}\n"
            "n_active     = {na} (expertos con f_i > 0)\n"
            "ratio        = {r}\n".format(
                n  = st.session_state.n_inferences,
                fi =", ".join(f"{v:.6f}" for v in fi_dbg),
                s  = float(fi_dbg.sum()),
                na = int((fi_dbg > 0).sum()),
                r  = f"{ratio:.6f}" if ratio is not None else "None",
            ),
            language="text",
        )

    if MOE_AVAILABLE and torch.cuda.is_available():
        st.divider()
        st.caption(f"GPU: `{torch.cuda.get_device_name(0)}`")
    elif MOE_AVAILABLE:
        st.caption("Dispositivo: CPU")

# ══════════════════════════════════════════════════════════════════════════════
#  COLUMNAS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 1.6], gap="large")

# ────────────────────────────────────────────────
# COLUMNA IZQUIERDA — Carga + Preprocesado
# ────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-title">1. Carga de Imagen</div>',
                unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Sube una o varias imágenes médicas (PNG, JPEG, NIfTI, NumPy)",
        type=["png", "jpg", "jpeg", "nii", "npy"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help=("Puedes subir múltiples archivos para inferencia en lote. "
              "Formatos: imágenes 2D (PNG/JPG) · volúmenes NIfTI (.nii) · arrays NumPy 3D (.npy)"),
    )

    # Firma del batch actual para detectar cambios en el uploader entre reruns
    current_sig = tuple((f.name, getattr(f, "size", None)) for f in (uploaded_files or []))
    if current_sig != st.session_state.get("batch_files_sig"):
        st.session_state.batch_files_sig = current_sig
        # Si el set de archivos cambió, descartamos los resultados del batch anterior
        # para no confundir al usuario con predicciones de una lista distinta.
        st.session_state.batch_results = []

    # Variables de estado de la imagen actual (la seleccionada para inspección)
    img_pil        = None
    orig_size      = None
    modality_label = None
    is_nifti_input = False
    nifti_volume   = None
    using_demo     = False
    uploaded_file  = None          # archivo actualmente seleccionado para preview
    selected_idx   = 0

    if not uploaded_files:
        st.info("Sube una o varias imágenes médicas para comenzar. Soporta PNG, JPEG, NIfTI (.nii) y NumPy (.npy).")
        # Imagen demo
        demo_arr = np.random.randint(60, 200, (224, 224, 3), dtype=np.uint8)
        demo_arr[80:140, 80:140] = [200, 220, 240]
        img_pil   = Image.fromarray(demo_arr)
        orig_size = (224, 224)
        modality_label = "2D (demo)"
        using_demo = True
        st.caption("_Vista previa con imagen de demostración_")
    else:
        using_demo = False
        n_files = len(uploaded_files)

        # Selector de archivo para previsualización e inspección detallada.
        # La inferencia se ejecuta igual sobre TODOS los archivos del batch.
        if n_files > 1:
            st.markdown(
                f'<span class="badge badge-blue">Batch · {n_files} archivos cargados</span>',
                unsafe_allow_html=True,
            )
            selected_idx = st.selectbox(
                "Imagen a inspeccionar en detalle",
                options=list(range(n_files)),
                format_func=lambda i: f"{i+1}. {uploaded_files[i].name}",
                key="batch_preview_idx",
                help="La inferencia se ejecuta sobre el batch completo; aquí eliges cuál ver en detalle.",
            )
        else:
            selected_idx = 0

        uploaded_file = uploaded_files[selected_idx]

        # ── Detección y carga via load_medical_input (nueva función) ─────
        #   Llama al wrapper nuevo. Toda la lógica de detección está allí.
        #   Las variables img_pil / nifti_volume / is_nifti_input / orig_size
        #   se asignan desde el resultado para mantener compatibilidad total
        #   con real_inference() y el bloque de preprocesado que sigue.
        med = load_medical_input(uploaded_file)

        if med["error"]:
            st.error(f"Error al cargar el archivo: {med['error']}")

        # ── Mapear campos del resultado a las variables existentes ─────────
        modality_label = med["modality"]
        orig_size      = med["orig_shape"]
        is_nifti_input = med["is_volumetric"]   # True para .npy 3D y .nii
        nifti_volume   = med["volume"]           # None si 2D
        img_pil        = med["preview_pil"]      # siempre un PIL Image

        # ── Visualización adaptada por tipo ───────────────────────────────
        if med["is_volumetric"] and med["volume"] is not None:
            volume    = med["volume"]
            n_slices  = med["n_slices"]
            file_type = med["file_type"]

            # Badge de formato
            fmt_label = "NIfTI" if file_type == "nifti" else "NumPy .npy"
            st.markdown(
                f'<span class="badge badge-blue">{fmt_label} · 3D · '
                f'{volume.shape[0]}×{volume.shape[1]}×{volume.shape[2]} voxels</span>',
                unsafe_allow_html=True,
            )

            # ── Selector de eje de visualización ──────────────────────────
            axis_names  = ["Axial (Z)", "Coronal (Y)", "Sagital (X)"]
            axis_limits = [volume.shape[2], volume.shape[1], volume.shape[0]]

            col_ax, col_sl = st.columns([1, 2])
            with col_ax:
                axis_sel = st.selectbox(
                    "Eje de visualización",
                    options=[0, 1, 2],
                    format_func=lambda i: axis_names[i],
                    key="vol_axis",
                )
            with col_sl:
                max_idx  = axis_limits[axis_sel] - 1
                def_idx  = max_idx // 2
                slice_idx = st.slider(
                    f"Slice ({axis_names[axis_sel]})",
                    min_value=0,
                    max_value=max_idx,
                    value=def_idx,
                    key="vol_slice_idx",
                    help=f"Navega los {max_idx + 1} slices del volumen en el eje seleccionado.",
                )

            # Extraer y mostrar el slice seleccionado
            sl_arr = _get_volume_slice(volume, axis=axis_sel, idx=slice_idx)
            sl_u8  = _normalize_slice_to_uint8(sl_arr)
            img_pil = Image.fromarray(sl_u8).convert("RGB")   # actualizar preview

            st.image(
                img_pil,
                caption=(f"{axis_names[axis_sel]} · slice {slice_idx}/{max_idx} — "
                         f"{uploaded_file.name}"),
                use_container_width=True,
            )

            # Mini-estadísticas del volumen
            with st.expander("Estadísticas del volumen"):
                st.caption(
                    f"**Shape:** {volume.shape}  · "
                    f"**Min:** {volume.min():.1f}  · "
                    f"**Max:** {volume.max():.1f}  · "
                    f"**Media:** {volume.mean():.1f}  · "
                    f"**Std:** {volume.std():.1f}"
                )

        else:
            # Imagen 2D — comportamiento idéntico al original
            st.image(
                img_pil,
                caption=f"Imagen cargada: {uploaded_file.name}",
                use_container_width=True,
            )

    # ── 2. Preprocesado Transparente ─────────────────────────────────────────
    st.markdown('<div class="section-title">2. Preprocesado Transparente</div>',
                unsafe_allow_html=True)

    if not using_demo and img_pil is not None:
        # Dimensiones adaptadas
        if is_nifti_input:
            adapt_size_str = "64×64×64"
            norm_type      = "HU clip [-1000, 400] → [0,1]"
            # Tomar una muestra del volumen (hasta 64 en cada eje)
            vol_s = nifti_volume
            slices = tuple(slice(0, min(64, s)) for s in vol_s.shape)
            vol_sample  = vol_s[slices]
            vol_clipped = np.clip(vol_sample, -1000, 400)
            vol_norm    = (vol_clipped - (-1000)) / (400 - (-1000))
            norm_mean, norm_std = float(vol_norm.mean()), float(vol_norm.std())
            norm_min,  norm_max = float(vol_norm.min()),  float(vol_norm.max())
        else:
            adapt_size_str = "224×224"
            norm_type      = "ImageNet (μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225])"
            arr_r  = np.array(img_pil.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
            mean_  = np.array([0.485, 0.456, 0.406])
            std_   = np.array([0.229, 0.224, 0.225])
            arr_n  = (arr_r - mean_) / std_
            norm_mean, norm_std = arr_n.mean(), arr_n.std()
            norm_min,  norm_max = arr_n.min(),  arr_n.max()

        # Tarjetas de métricas de preprocesado
        if is_nifti_input:
            # 3D: orig_size es una tupla (D, H, W)
            if len(orig_size) >= 3:
                orig_str = f"{orig_size[0]}×{orig_size[1]}×{orig_size[2]}"
            else:
                orig_str = "×".join(str(d) for d in orig_size)
        else:
            # 2D: PIL devuelve (W, H); numpy puede ser (H, W) o (H, W, C)
            if len(orig_size) >= 2:
                orig_str = f"{orig_size[0]}×{orig_size[1]}"
                if len(orig_size) == 3:
                    orig_str += f"×{orig_size[2]}"
            else:
                orig_str = str(orig_size)

        cols_pre = st.columns(3)
        with cols_pre[0]:
            st.markdown(f"""<div class="metric-card">
                <h3 style="font-size:1.2em">{orig_str}</h3>
                <p>Dimensiones originales</p></div>""", unsafe_allow_html=True)
        with cols_pre[1]:
            st.markdown(f"""<div class="metric-card">
                <h3 style="font-size:1.2em">{adapt_size_str}</h3>
                <p>Dimensiones adaptadas</p></div>""", unsafe_allow_html=True)
        with cols_pre[2]:
            st.markdown(f"""<div class="metric-card">
                <h3 style="font-size:1.2em">{modality_label}</h3>
                <p>Modalidad detectada</p></div>""", unsafe_allow_html=True)

        with st.expander("Ver estadísticas del tensor normalizado"):
            st.code(f"""
Modalidad : {modality_label}
Norma     : {norm_type}
Forma out : {'(1, 1, 64, 64, 64)' if is_nifti_input else '(1, 3, 224, 224)'}
Media     : {norm_mean:.4f}
Std       : {norm_std:.4f}
Min / Max : {norm_min:.3f} / {norm_max:.3f}
dtype     : float32
            """, language="text")
    else:
        st.caption("_Carga una imagen real para ver las dimensiones de preprocesado._")

    # ── Botón de inferencia ───────────────────────────────────────────────────
    st.divider()
    n_batch = 0 if using_demo or not uploaded_files else len(uploaded_files)
    btn_label = ("Ejecutar Inferencia" if n_batch <= 1
                 else f"Ejecutar Inferencia sobre {n_batch} imágenes")
    btn_help  = ("Sube una imagen primero" if using_demo
                 else (f"Pipeline MoE sobre {n_batch} archivo(s)" if n_batch else
                       "Ejecutar pipeline MoE"))
    run_inference = st.button(
        btn_label,
        type="primary",
        use_container_width=True,
        disabled=using_demo,
        help=btn_help,
    )

# ────────────────────────────────────────────────
# COLUMNA DERECHA — Resultados
# ────────────────────────────────────────────────
with col_right:
    show_results = (run_inference or
                    (st.session_state.n_inferences > 0
                     and st.session_state.last_result is not None))

    if show_results:

        # ── Ejecutar inferencia nueva ─────────────────────────────────────
        if run_inference and not using_demo:
            batch = uploaded_files or []
            n     = len(batch)
            batch_results_local = []

            progress = st.progress(0.0, text=f"Procesando {n} imagen(es)…")

            for i, upl in enumerate(batch):
                progress.progress(i / max(n, 1),
                                  text=f"[{i+1}/{n}] {upl.name}")
                try:
                    upl.seek(0)   # rebobinar stream por si ya fue leído
                except Exception:
                    pass

                med_i = load_medical_input(upl)
                res_i = real_inference(
                    med_i["preview_pil"],
                    med_i["is_volumetric"],
                    med_i["volume"],
                )

                # Chequeo OOD combinado (entropía + coseno mínimo)
                ood_i = detect_ood(res_i, ood_threshold, ood_cos_threshold)

                # Contabilizar en load balance solo si NO es OOD (o si el bloqueo está off).
                # Así las imágenes rechazadas no contaminan f_i.
                if not (ood_i["is_ood"] and block_ood_inference):
                    update_load_balance(res_i["expert_idx"])

                entry = {
                    "filename":    upl.name,
                    "modality":    med_i["modality"],
                    "expert_idx":  res_i["expert_idx"],
                    "pred_label":  res_i["pred_label"],
                    "confidence":  res_i["confidence"],
                    "entropy":     res_i["entropy"],
                    "latency_ms":  res_i["latency_ms"],
                    "mode":        res_i.get("mode", "demo"),
                    "is_ood":      ood_i["is_ood"],
                    "ood_reasons": ood_i["reasons"],
                    "max_cos":     ood_i["max_cos"],
                    "result":      res_i,
                    "preview_pil": med_i["preview_pil"],
                    "is_nifti":    med_i["is_volumetric"],
                    "volume":      med_i["volume"],
                }
                batch_results_local.append(entry)

            progress.progress(1.0, text=f"Batch completado ({n} imagen(es))")
            time.sleep(0.2)
            progress.empty()

            # Guardar en session_state: batch completo + último resultado
            st.session_state.batch_results = batch_results_local
            # last_result apunta al resultado del archivo actualmente seleccionado
            # en el selectbox para preservar la UX individual existente.
            sel = selected_idx if selected_idx < len(batch_results_local) else 0
            st.session_state.last_result = batch_results_local[sel]["result"]

            # Rerun completo para que el sidebar (Debug Load Balance y
            # caption de ratio) lea el session_state actualizado.
            st.rerun()

        # ── Lectura del resultado a mostrar en detalle ────────────────────
        #   Si el usuario cambió el selectbox después de un batch, tomamos
        #   el resultado correspondiente de batch_results; si no hay batch,
        #   recurrimos a last_result (flujo individual histórico).
        result = None
        batch_results = st.session_state.batch_results
        if batch_results and 0 <= selected_idx < len(batch_results):
            result = batch_results[selected_idx]["result"]
            # Mantener last_result sincronizado con la selección actual
            st.session_state.last_result = result
        else:
            result = st.session_state.last_result

        if result is None:
            st.info("Sube una o más imágenes y presiona **Ejecutar Inferencia**.")
            st.stop()

        # Badge de modo
        mode = result.get("mode", "demo")
        if mode == "real":
            st.markdown('<span class="badge badge-green">Inferencia REAL</span>',
                        unsafe_allow_html=True)
        else:
            reason = result.get("demo_reason", "")
            st.markdown(f'<span class="badge badge-orange">Modo DEMO</span>'
                        f'<small style="color:#856404">{reason[:80]}</small>',
                        unsafe_allow_html=True)

        # ── Chequeo OOD sobre el resultado actualmente seleccionado ───────
        #   Se recalcula en cada render para que los cambios de umbral en la
        #   sidebar actualicen el banner sin tener que re-ejecutar inferencia.
        ood_now = detect_ood(result, ood_threshold, ood_cos_threshold)
        ood_blocked = ood_now["is_ood"] and block_ood_inference

        if ood_now["is_ood"]:
            st.markdown(f"""
            <div class="ood-alert" style="border-color:#dc3545; background:#f8d7da;">
            <strong>IMAGEN FUERA DE DISTRIBUCIÓN (OOD)</strong><br>
            <small>Motivo: {ood_now["reasons"]}.</small><br>
            <small>Coseno al vecino más cercano: <code>{ood_now["max_cos"]:.3f}</code>
            · Entropía H(g): <code>{ood_now["entropy"]:.3f}</code>.</small><br>
            {"<strong>El experto NO recibió esta imagen.</strong> El router la rechazó antes de la delegación."
              if ood_blocked else
              "<small><em>Bloqueo desactivado: el experto procesó la imagen de todos modos. "
              "Revisa el resultado con cautela.</em></small>"}
            </div>
            """, unsafe_allow_html=True)

        exp_info = EXPERTS_UI[result["expert_idx"]]

        # Si está bloqueado por OOD, no mostramos predicción del experto.
        # El flujo sigue con el heatmap y los detalles del router para
        # que el usuario vea POR QUÉ fue rechazada.
        if ood_blocked:
            st.info(
                "Las secciones de predicción del experto están suprimidas "
                "porque la imagen se rechazó. Puedes desactivar el bloqueo en "
                "la barra lateral si quieres forzar la inferencia del experto."
            )

        # ── 3. Inferencia en tiempo real ──────────────────────────────────
        # Suprimida cuando la imagen se rechazó por OOD: no tiene sentido
        # mostrar la predicción de un experto que nunca procesó la imagen.
        if not ood_blocked:
            st.markdown('<div class="section-title">3. Inferencia en Tiempo Real</div>',
                        unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="metric-card">
                    <h3 style="font-size:1.05em">{result['pred_label']}</h3>
                    <p>Predicción</p></div>""", unsafe_allow_html=True)
            with c2:
                conf_pct   = result["confidence"] * 100
                conf_color = "#28a745" if conf_pct > 70 else "#ffc107"
                st.markdown(f"""<div class="metric-card">
                    <h3 style="color:{conf_color}">{conf_pct:.1f}%</h3>
                    <p>Confianza</p></div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                    <h3>{result['latency_ms']:.1f} ms</h3>
                    <p>Tiempo inferencia</p></div>""", unsafe_allow_html=True)

            # Barra de probabilidades por clase
            with st.expander("Ver distribución de probabilidades por clase"):
                labels_exp = result.get("labels", LABELS_UI.get(result["expert_idx"], []))
                probs_exp  = result["probs"]
                fig_pb = go.Figure(go.Bar(
                    x=probs_exp,
                    y=labels_exp,
                    orientation="h",
                    marker_color=["#2d6a9f" if i == result["pred_idx"] else "#a8c8e8"
                                  for i in range(len(probs_exp))],
                    text=[f"{p:.3f}" for p in probs_exp],
                    textposition="outside",
                ))
                fig_pb.update_layout(
                    height=max(180, len(labels_exp) * 30),
                    margin=dict(t=10, b=10, l=10, r=60),
                    xaxis=dict(range=[0, 1.1]),
                    plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc",
                )
                st.plotly_chart(fig_pb, use_container_width=True)

        # ── 4. Attention Heatmap ──────────────────────────────────────────
        st.markdown('<div class="section-title">4. Attention Heatmap — Router ViT</div>',
                    unsafe_allow_html=True)

        overlay = overlay_heatmap(img_pil, result["attn_map"])

        fig_heat, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        fig_heat.patch.set_facecolor("#f8fafc")
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])

        axes[0].imshow(img_pil.convert("RGB").resize((224, 224)))
        axes[0].set_title("Original", fontsize=9, fontweight="bold")

        im = axes[1].imshow(result["attn_map"], cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Attention Map (CLS→patches)", fontsize=9, fontweight="bold")
        fig_heat.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay (α=0.45)", fontsize=9, fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig_heat, use_container_width=True)
        plt.close(fig_heat)

        # Info del método de heatmap
        if mode == "real":
            st.caption("_Heatmap generado desde attention rollout del último bloque ViT"
                       "(CLS token → patches, promedio de heads)._")
        else:
            st.caption("_Heatmap simulado (demo). Conecta los checkpoints para ver la atención real._")

        # ── 5. Panel del Experto Activado ─────────────────────────────────
        # Suprimido cuando está bloqueado por OOD: el experto no recibió la
        # imagen, así que mostrar su nombre "activado" sería engañoso.
        # El detalle del router k-NN SÍ se muestra porque explica por qué
        # la imagen fue rechazada (vecinos lejanos, coseno bajo).
        router_info = result.get("router_info", {})
        if not ood_blocked:
            st.markdown('<div class="section-title">5. Experto Activado</div>',
                        unsafe_allow_html=True)

            gating_scores = result["gating"]
            confidence_r  = router_info.get("confidence", gating_scores[result["expert_idx"]])

            st.markdown(f"""
            <div class="expert-card">
              <h4>{exp_info['name']}</h4>
              <p><strong>Arquitectura:</strong> {exp_info['arch']}</p>
              <p><strong>Dataset origen:</strong> {exp_info['dataset']}</p>
              <p><strong>Tarea:</strong> {exp_info['task']}</p>
              <p><strong>Gating score:</strong>
                 <code>{gating_scores[result['expert_idx']]:.4f}</code>
                 &nbsp;|&nbsp;
                 <strong>Router conf.:</strong> <code>{float(confidence_r):.2%}</code>
                 &nbsp;<span class="badge badge-blue">k-NN FAISS</span></p>
            </div>
            """, unsafe_allow_html=True)

            # Gating bar chart
            fig_gate = go.Figure(go.Bar(
                x=[EXPERTS_UI[i]["name"].split("—")[1].strip() for i in range(5)],
                y=gating_scores,
                marker_color=["#2d6a9f" if i == result["expert_idx"] else "#a8c8e8"
                              for i in range(5)],
                text=[f"{g:.3f}" for g in gating_scores],
                textposition="outside",
            ))
            fig_gate.update_layout(
                title="Gating scores — todos los expertos",
                yaxis=dict(range=[0, 1.15], title="Score"),
                height=230, margin=dict(t=40, b=30, l=30, r=10),
                plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc",
            )
            st.plotly_chart(fig_gate, use_container_width=True)

        # Detalle k-NN si está disponible (siempre visible, ayuda a justificar el OOD)
        if router_info and router_info.get("cosine_scores"):
            expander_label = ("Ver vecinos del router k-NN (razón del rechazo OOD)"
                              if ood_blocked else "Ver detalle del router k-NN")
            with st.expander(expander_label, expanded=ood_blocked):
                cols_r = st.columns(2)
                with cols_r[0]:
                    st.markdown("**Vecinos más cercanos (k=5):**")
                    nbr_labels = router_info.get("neighbor_labels", [])
                    nbr_scores = router_info.get("cosine_scores",   [])
                    for j, (lbl, sc) in enumerate(zip(nbr_labels, nbr_scores)):
                        expert_name = EXPERTS_UI.get(int(lbl), {}).get("name", f"Exp{lbl}")
                        st.caption(f"  {j+1}. {expert_name} — cos={sc:.4f}")
                with cols_r[1]:
                    st.markdown("**Votos por experto:**")
                    votes = router_info.get("vote_counts", {})
                    for eid, cnt in sorted(votes.items()):
                        st.caption(f"  Exp{int(eid)+1}: {cnt} voto(s)")

        # ── 6. Resumen del Batch (cuando hay más de una imagen) ───────────
        batch_results = st.session_state.batch_results
        if len(batch_results) > 1:
            st.markdown(
                '<div class="section-title">Resumen del Batch</div>',
                unsafe_allow_html=True,
            )

            df_batch = pd.DataFrame([
                {
                    "#":           i + 1,
                    "Archivo":     entry["filename"],
                    "Modalidad":   entry["modality"],
                    "OOD":         "Rechazada" if entry.get("is_ood") else "OK",
                    "Experto":     ("— (bloqueado)" if entry.get("is_ood") and block_ood_inference
                                     else EXPERTS_UI[entry["expert_idx"]]["name"]),
                    "Predicción":  ("—" if entry.get("is_ood") and block_ood_inference
                                     else entry["pred_label"]),
                    "Cos máx":     f"{entry.get('max_cos', 0.0):.3f}",
                    "Confianza":   f"{entry['confidence']*100:.1f}%",
                    "Entropía":    f"{entry['entropy']:.3f}",
                    "Latencia ms": f"{entry['latency_ms']:.1f}",
                    "Modo":        entry["mode"],
                }
                for i, entry in enumerate(batch_results)
            ])

            # Resaltar la fila del archivo seleccionado en el selector
            def _hl_selected(row):
                return (["background-color:#d4edda; font-weight:600"] * len(row)
                        if row["#"] == selected_idx + 1
                        else [""] * len(row))

            st.dataframe(
                df_batch.style.apply(_hl_selected, axis=1),
                use_container_width=True, hide_index=True,
            )

            # Métricas agregadas (solo imágenes no-OOD para confianza)
            c_b1, c_b2, c_b3 = st.columns(3)
            n_ok = sum(1 for e in batch_results if not e.get("is_ood"))
            with c_b1:
                if n_ok > 0:
                    avg_conf = np.mean([e["confidence"] for e in batch_results
                                        if not e.get("is_ood")]) * 100
                    st.markdown(f"""<div class="metric-card">
                        <h3>{avg_conf:.1f}%</h3>
                        <p>Confianza media ({n_ok} aceptadas)</p></div>""",
                        unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="metric-card">
                        <h3>—</h3>
                        <p>Todas rechazadas por OOD</p></div>""", unsafe_allow_html=True)
            with c_b2:
                avg_lat = np.mean([e["latency_ms"] for e in batch_results])
                st.markdown(f"""<div class="metric-card">
                    <h3>{avg_lat:.1f} ms</h3>
                    <p>Latencia media</p></div>""", unsafe_allow_html=True)
            with c_b3:
                n_ood = sum(1 for e in batch_results if e.get("is_ood"))
                n_tot = len(batch_results)
                pct_ood = (n_ood / n_tot * 100) if n_tot else 0.0
                st.markdown(f"""<div class="metric-card">
                    <h3>{n_ood} / {n_tot}</h3>
                    <p>OOD detectados ({pct_ood:.0f} %)</p></div>""",
                    unsafe_allow_html=True)

            # Distribución de expertos — solo las aceptadas
            exp_counts = np.zeros(5, dtype=int)
            for e in batch_results:
                if not (e.get("is_ood") and block_ood_inference):
                    exp_counts[e["expert_idx"]] += 1

            fig_dist = go.Figure(go.Bar(
                x=[EXPERTS_UI[i]["name"].split("—")[1].strip() for i in range(5)],
                y=exp_counts,
                marker_color=["#2d6a9f" if c > 0 else "#d3d3d3" for c in exp_counts],
                text=[str(c) if c > 0 else "" for c in exp_counts],
                textposition="outside",
            ))
            fig_dist.update_layout(
                title=f"Distribución de enrutamiento · {len(batch_results)} imagen(es)",
                yaxis=dict(title="Imágenes enrutadas", rangemode="tozero"),
                height=240, margin=dict(t=40, b=30, l=30, r=10),
                plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc",
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # Descarga CSV del resumen
            csv_bytes = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar resumen del batch (CSV)",
                data=csv_bytes,
                file_name="batch_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

    else:
        st.info("Sube una o varias imágenes y presiona **Ejecutar Inferencia** para ver los resultados.")

# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN INFERIOR — Ablation, Load Balance, OOD
# ══════════════════════════════════════════════════════════════════════════════
st.divider()

tabs_lower = []
if show_ablation: tabs_lower.append("6. Ablation Study")
if show_balance:  tabs_lower.append("7. Load Balance")
if show_ood:      tabs_lower.append("8. OOD Detection")

if tabs_lower:
    tabs    = st.tabs(tabs_lower)
    tab_idx = 0

    # ── 6. Ablation Study ────────────────────────────────────────────────────
    if show_ablation:
        with tabs[tab_idx]:
            st.markdown("#### Comparativa de los 4 Mecanismos de Routing")

            abl = load_ablation_data()
            src = abl.pop("_source", "hardcoded")
            src_badge = "desde JSON" if src == "json" else "datos integrados"
            st.caption(f"Fuente de datos: **{src_badge}** — {ABLATION_JSON}")

            df_abl = pd.DataFrame(abl)

            # Identificar ganador (mayor Routing Acc.)
            winner_idx = int(np.argmax(abl["Routing Acc."]))
            winner_router = abl["Router"][winner_idx]

            # Formatear para display
            df_display = df_abl.copy()
            df_display["Routing Acc."]  = df_display["Routing Acc."].map("{:.0%}".format)
            df_display["Latencia (ms)"] = df_display["Latencia (ms)"].map("{:.1f}".format)
            df_display["VRAM (MB)"]     = df_display["VRAM (MB)"].map("{:,}".format)

            st.dataframe(
                df_display.style.apply(
                    lambda row: ["background-color:#d4edda; font-weight:bold"] * len(row)
                    if row["Router"] == winner_router else [""] * len(row),
                    axis=1,
                ),
                use_container_width=True, hide_index=True,
            )

            col_a1, col_a2 = st.columns(2)

            with col_a1:
                colors_bar = ["#2d6a9f", "#4caf50", "#ff9800", "#9c27b0"]
                fig_acc = go.Figure(go.Bar(
                    x=abl["Router"],
                    y=[v * 100 for v in abl["Routing Acc."]],
                    marker_color=colors_bar[:len(abl["Router"])],
                    text=[f"{v*100:.0f}%" for v in abl["Routing Acc."]],
                    textposition="outside",
                ))
                fig_acc.add_hline(y=80, line_dash="dash", line_color="red",
                                  annotation_text="Umbral 80%")
                fig_acc.update_layout(
                    title="Routing Accuracy por método",
                    yaxis=dict(range=[0, 115], title="Accuracy (%)"),
                    height=320, margin=dict(t=45, b=80, l=40, r=10),
                    plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc",
                    xaxis=dict(tickangle=-20),
                )
                st.plotly_chart(fig_acc, use_container_width=True)

            with col_a2:
                fig_scat = go.Figure()
                for i, router in enumerate(abl["Router"]):
                    fig_scat.add_trace(go.Scatter(
                        x=[float(abl["Latencia (ms)"][i])],
                        y=[abl["Routing Acc."][i] * 100],
                        mode="markers+text",
                        name=router,
                        text=[router.split(" +")[0]],
                        textposition="top center",
                        marker=dict(size=16, color=colors_bar[i % len(colors_bar)]),
                    ))
                fig_scat.update_layout(
                    title="Latencia vs Routing Accuracy",
                    xaxis_title="Latencia (ms)", yaxis_title="Accuracy (%)",
                    height=320, margin=dict(t=45, b=40, l=40, r=10),
                    plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc",
                    showlegend=False,
                )
                st.plotly_chart(fig_scat, use_container_width=True)

            st.success(
                f"**Router ganador:** `{winner_router}` con "
                f"**{abl['Routing Acc.'][winner_idx]:.0%}** Routing Accuracy."
            )
        tab_idx += 1

    # ── 7. Load Balance ──────────────────────────────────────────────────────
    if show_balance:
        with tabs[tab_idx]:
            st.markdown("#### Load Balance — Distribución Acumulada de Enrutamiento")
            fi              = st.session_state.f_i_history
            ratio, n_active = compute_load_ratio()
            n_inf           = st.session_state.n_inferences

            col_lb1, col_lb2 = st.columns([1.6, 1])
            with col_lb1:
                bar_colors = ["#2d6a9f" if fi[i] == fi.max() else "#a8c8e8"
                              for i in range(5)]
                fig_lb = go.Figure(go.Bar(
                    x=[EXPERTS_UI[i]["name"].split("—")[1].strip() for i in range(5)],
                    y=fi * 100,
                    marker_color=bar_colors,
                    text=[f"{v:.1%}" for v in fi],
                    textposition="outside",
                ))
                fig_lb.add_hline(y=20, line_dash="dot", line_color="gray",
                                 annotation_text="Balance perfecto (20%)")
                fig_lb.add_hline(y=20 * balance_limit, line_dash="dash",
                                 line_color="red",
                                 annotation_text=f"Límite ({fmt_es(balance_limit, 2)}×)")
                fig_lb.update_layout(
                    title=f"f_i acumulado — {n_inf} inferencia(s)",
                    yaxis=dict(range=[0, 65], title="f_i (%)"),
                    height=340, margin=dict(t=50, b=40, l=40, r=10),
                    plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc",
                )
                st.plotly_chart(fig_lb, use_container_width=True)

            with col_lb2:
                st.markdown("**Detalle de carga:**")
                for i in range(5):
                    pct      = fi[i] * 100
                    bar_html = f"{'█' * int(pct // 4)}{'░' * (25 - int(pct // 4))}"
                    st.markdown(f"**Exp{i+1}** `{bar_html}` {pct:.1f}%")
                st.divider()
                if ratio is None:
                    st.metric("Cociente max/min", "—",
                              delta=f"Límite: {fmt_es(balance_limit, 2)}")
                    st.info(
                        "Aún no hay inferencias. Carga al menos una imagen "
                        "para comenzar a medir el balance de carga."
                    )
                elif n_active < 5:
                    missing = 5 - n_active
                    st.metric(
                        f"Cociente max/min ({n_active}/5 activos)",
                        fmt_es(ratio, 3),
                        delta=f"Límite: {fmt_es(balance_limit, 2)}",
                    )
                    st.warning(
                        f"**Cobertura parcial** — el cociente se calcula sobre "
                        f"los **{n_active}** experto(s) que han recibido "
                        f"inferencias. Faltan **{missing}** por activar para "
                        f"evaluar el balance completo sobre los 5 dominios."
                    )
                else:
                    st.metric("Cociente max/min", fmt_es(ratio, 3),
                              delta=f"Límite: {fmt_es(balance_limit, 2)}")
                    if ratio > balance_limit:
                        st.error(f"**PENALIZACIÓN ACTIVA** — cociente "
                                 f"{fmt_es(ratio, 3)} > {fmt_es(balance_limit, 2)}\n\nAjusta α en Auxiliary Loss.")
                    else:
                        st.success("Balance dentro del límite permitido.")
                st.caption("_La penalización aplica solo al router ViT+Linear y requiere los 5 expertos activos._")
        tab_idx += 1

    # ── 8. OOD Detection ─────────────────────────────────────────────────────
    if show_ood:
        with tabs[tab_idx]:
            st.markdown("#### OOD Detection — Detección por Entropía del Gating Score")
            st.caption(
                "La entropía H(g) del vector de gating actúa como señal de incertidumbre. "
                "Alta entropía → el router no reconoce la imagen → posible OOD."
            )

            col_o1, col_o2 = st.columns([1, 1])
            with col_o1:
                if st.session_state.last_result is not None:
                    entropy = st.session_state.last_result["entropy"]
                    gating  = st.session_state.last_result["gating"]
                    is_ood  = entropy > ood_threshold
                    max_ent = np.log(5)

                    st.markdown("**Última imagen procesada:**")
                    if is_ood:
                        st.markdown(f"""
                        <div class="ood-alert">
                        <strong>POSIBLE OOD DETECTADO</strong><br>
                        Entropía H(g) = <strong>{entropy:.4f}</strong>
                        (umbral: {ood_threshold:.2f})<br>
                        Esta imagen podría no pertenecer a ningún dominio conocido.
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="ood-ok">
                        <strong>Imagen IN-DISTRIBUTION</strong><br>
                        Entropía H(g) = <strong>{entropy:.4f}</strong>
                        (umbral: {ood_threshold:.2f})<br>
                        El router reconoce la imagen con buena confianza.
                        </div>""", unsafe_allow_html=True)

                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=entropy,
                        delta={"reference": ood_threshold,
                               "increasing": {"color": "#dc3545"}},
                        number={"suffix":" nats", "font": {"size": 22}},
                        gauge={
                            "axis": {"range": [0, max_ent]},
                            "bar":  {"color": "#dc3545" if is_ood else "#28a745"},
                            "steps": [
                                {"range": [0, ood_threshold],      "color": "#d4edda"},
                                {"range": [ood_threshold, max_ent],"color": "#f8d7da"},
                            ],
                            "threshold": {
                                "line":      {"color": "orange", "width": 3},
                                "thickness": 0.75,
                                "value":     ood_threshold,
                            },
                        },
                        title={"text": "Entropía H(g)"},
                    ))
                    fig_gauge.update_layout(height=280,
                                            margin=dict(t=40, b=10, l=20, r=20),
                                            paper_bgcolor="#f8fafc")
                    st.plotly_chart(fig_gauge, use_container_width=True)
                else:
                    st.info("Ejecuta una inferencia para ver el estado OOD.")

            with col_o2:
                st.markdown("**Distribución de entropías de referencia:**")
                np.random.seed(42)
                ent_in  = np.random.beta(2, 5, 200) * np.log(5)
                ent_ood = np.random.beta(5, 2, 80)  * np.log(5)

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=ent_in, name="In-Distribution",
                    marker_color="#28a745", opacity=0.7, nbinsx=25))
                fig_hist.add_trace(go.Histogram(
                    x=ent_ood, name="OOD",
                    marker_color="#dc3545", opacity=0.7, nbinsx=25))
                fig_hist.add_vline(x=ood_threshold, line_dash="dash",
                                   line_color="orange",
                                   annotation_text=f"Umbral {ood_threshold:.2f}")
                fig_hist.update_layout(
                    barmode="overlay",
                    title="Histograma H(g) — sistema calibrado",
                    xaxis_title="Entropía (nats)", yaxis_title="Frecuencia",
                    height=300, margin=dict(t=50, b=40, l=40, r=10),
                    plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                st.caption("""
                **Fórmula:** H(g) = −∑ gᵢ · log(gᵢ)  
                **Máx. posible:** log(5) ≈ 1.609 nats  
                **Calibración:** Ajustar umbral con imágenes fuera de dominio.
                """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
mode_str = "Pipeline MoE real conectado" if (MOE_AVAILABLE and load_moe_model()[0] is not None) \
           else "Modo demo — conecta los checkpoints"
st.markdown(
    f"<center><small><b>MoE Medical Dashboard</b> — Proyecto Final Visión · "
    f"Incorporar Elementos de IA, Unidad II · <code>{mode_str}</code></small></center>",
    unsafe_allow_html=True,
)
