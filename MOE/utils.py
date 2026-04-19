"""
utils.py — Utilidades del sistema MoE
Logging, formateo de resultados, persistencia JSON, métricas de tiempo.
"""

import json
import time
import logging
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


# ── Logger ────────────────────────────────────────────────────────────────────

def setup_logger(
    name:     str  = "moe",
    level:    int  = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """Configura y devuelve un logger con formato limpio."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        if log_file:
            fh = logging.FileHandler(str(log_file))
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


logger = setup_logger()


# ── Timer ─────────────────────────────────────────────────────────────────────

class Timer:
    """Context manager para medir tiempos de ejecución."""

    def __init__(self, label: str = ""):
        self.label   = label
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if self.label:
            logger.info(f"[Timer] {self.label}: {self.elapsed*1000:.1f} ms")

    def __str__(self):
        return f"{self.elapsed*1000:.1f} ms"


# ── Formato de resultados ─────────────────────────────────────────────────────

EXPERT_NAMES = {
    0: "ChestX-ray14 (NIH)",
    1: "Osteoartritis (KL Grade)",
    2: "Dermoscopia (ISIC 2019)",
    3: "Nódulos Pulmonares (LUNA16)",
    4: "Páncreas CT (Segmentación)",
}

EXPERT_LABELS = {
    0: ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule"],
    1: ["KL-0 (Normal)", "KL-1 (Dudoso)", "KL-2 (Leve)", "KL-3 (Moderado)", "KL-4 (Severo)"],
    2: ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC"],
    3: ["Negativo", "Nódulo Maligno"],
    4: ["Normal", "Patología"],
}


def format_prediction(
    output:      torch.Tensor,
    expert_id:   int,
    router_info: dict,
    elapsed_ms:  float,
) -> dict:
    """
    Formatea la predicción del MoE en un dict estructurado y legible.
    """
    probs = output[0].detach().cpu().float().numpy()
    labels = EXPERT_LABELS.get(expert_id, [f"class_{i}" for i in range(len(probs))])

    # Construir scores por clase
    class_scores = {
        label: float(round(float(p), 4))
        for label, p in zip(labels, probs)
    }

    # Predicción principal
    pred_idx   = int(np.argmax(probs))
    pred_label = labels[pred_idx] if pred_idx < len(labels) else f"class_{pred_idx}"
    confidence = float(round(float(probs[pred_idx]), 4))

    result = {
        "timestamp":     datetime.now().isoformat(),
        "expert_id":     expert_id,
        "expert_name":   EXPERT_NAMES.get(expert_id, f"Expert {expert_id}"),
        "modality":      router_info.get("modality", "unknown"),
        "prediction": {
            "class_index": pred_idx,
            "class_label": pred_label,
            "confidence":  confidence,
        },
        "class_scores":  class_scores,
        "router": {
            "k_neighbors":    router_info.get("neighbor_labels", []),
            "cosine_scores":  [round(s, 4) for s in router_info.get("cosine_scores", [])],
            "vote_counts":    router_info.get("vote_counts", {}),
            "router_confidence": round(router_info.get("confidence", 0.0), 4),
        },
        "inference_time_ms": round(elapsed_ms, 2),
    }

    return result


def print_prediction(result: dict) -> None:
    """Imprime el resultado de forma legible en consola."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  MoE — Resultado de Inferencia")
    print(sep)
    print(f"  Experto:      [{result['expert_id']}] {result['expert_name']}")
    print(f"  Modalidad:    {result['modality']}")
    print(f"  Predicción:   {result['prediction']['class_label']}")
    print(f"  Confianza:    {result['prediction']['confidence']*100:.1f}%")
    print(f"  Tiempo:       {result['inference_time_ms']} ms")
    print(f"\n  Distribución de clases:")
    for cls, score in result["class_scores"].items():
        bar = "█" * int(score * 20)
        print(f"    {cls:<30} {score:.3f}  {bar}")
    print(f"\n  Router (k-NN):")
    print(f"    Votos:      {result['router']['vote_counts']}")
    print(f"    Confianza:  {result['router']['router_confidence']*100:.0f}%")
    print(sep + "\n")


# ── Persistencia JSON ─────────────────────────────────────────────────────────

def save_result_json(
    result:    dict,
    output_path: Union[str, Path],
    append:    bool = True,
) -> Path:
    """
    Guarda el resultado en un archivo JSON.
    Si append=True, añade al array existente (historial).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if append and out.exists():
        with open(out, "r") as f:
            history = json.load(f)
        if not isinstance(history, list):
            history = [history]
    else:
        history = []

    history.append(result)

    with open(out, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    logger.info(f"Resultado guardado en {out} ({len(history)} entradas)")
    return out


def load_results_json(path: Union[str, Path]) -> List[dict]:
    """Carga historial de resultados desde JSON."""
    with open(str(path), "r") as f:
        return json.load(f)


# ── Device utils ──────────────────────────────────────────────────────────────

def get_device(prefer_gpu: bool = True) -> str:
    """Devuelve 'cuda' si disponible y prefer_gpu=True, sino 'cpu'."""
    if prefer_gpu and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detectada: {gpu_name}")
        return "cuda"
    logger.info("Usando CPU")
    return "cpu"


def memory_summary(device: str = "cuda") -> str:
    """Resumen de memoria GPU."""
    if not torch.cuda.is_available():
        return "GPU no disponible"
    alloc = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"GPU Mem: {alloc:.2f} GB / {total:.2f} GB"


# ── Numpy/Torch helpers ───────────────────────────────────────────────────────

def to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.array(x, dtype=np.float32)


def to_tensor(x: Union[np.ndarray, list], device: str = "cpu") -> torch.Tensor:
    return torch.tensor(np.array(x), dtype=torch.float32, device=device)
