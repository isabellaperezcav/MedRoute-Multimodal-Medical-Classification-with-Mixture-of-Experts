"""
inference.py — Script de inferencia MoE
Uso:
  python -m moe.inference --input imagen.png
  python -m moe.inference --input volumen.nii.gz --device cuda --fp16
  python -m moe.inference --input carpeta/ --batch --output resultados.json
"""

import argparse
import sys
import time
import json
from pathlib import Path

import torch

# Permitir ejecución tanto como módulo como script directo
try:
    from .moe_model import MoEModel
    from .utils     import (format_prediction, print_prediction,
                             save_result_json, get_device, Timer, logger)
except ImportError:
    # Ejecución directa: python inference.py
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from moe.moe_model import MoEModel
    from moe.utils     import (format_prediction, print_prediction,
                                save_result_json, get_device, Timer, logger)


# ── Extensiones soportadas ────────────────────────────────────────────────────

SUPPORTED_2D = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
SUPPORTED_3D = {".nii", ".gz", ".npy"}
SUPPORTED    = SUPPORTED_2D | SUPPORTED_3D


def is_supported(path: Path) -> bool:
    suffix = "".join(path.suffixes).lower()
    return any(suffix.endswith(ext) for ext in SUPPORTED)


# ── Inferencia individual ─────────────────────────────────────────────────────

def run_single(
    model:   MoEModel,
    path:    Path,
    verbose: bool = True,
) -> dict:
    """Inferencia sobre un único archivo."""
    logger.info(f"Procesando: {path}")

    with Timer("Inferencia total") as t:
        output, expert_id, router_info = model.predict_from_file(path)

    result = format_prediction(output, expert_id, router_info,
                                elapsed_ms=t.elapsed * 1000)
    result["input_file"] = str(path)

    if verbose:
        print_prediction(result)

    return result


# ── Inferencia batch ──────────────────────────────────────────────────────────

def run_batch(
    model:   MoEModel,
    folder:  Path,
    verbose: bool = True,
) -> list:
    """
    Inferencia sobre todos los archivos soportados en una carpeta.
    El batch es secuencial (B=1 por imagen) porque cada imagen puede
    ir a un experto diferente.
    """
    files = [f for f in sorted(folder.rglob("*")) if is_supported(f)]
    if not files:
        logger.warning(f"No se encontraron archivos soportados en {folder}")
        return []

    logger.info(f"Batch: {len(files)} archivos en {folder}")
    results = []

    for i, f in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] {f.name}")
        try:
            r = run_single(model, f, verbose=verbose)
            results.append(r)
        except Exception as e:
            logger.error(f"  Error en {f.name}: {e}")
            results.append({"input_file": str(f), "error": str(e)})

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MoE Medical Inference System",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument(
        "--input", "-i", required=True,
        help="Ruta a imagen (.png/.jpg), volumen (.nii/.nii.gz) "
             "o carpeta (modo batch)"
    )
    p.add_argument(
        "--base_dir", "-b", default=".",
        help="Directorio raíz del proyecto (default: directorio actual)"
    )
    p.add_argument(
        "--device", "-d", default=None,
        choices=["cpu", "cuda"],
        help="Dispositivo (default: auto-detectar)"
    )
    p.add_argument(
        "--fp16", action="store_true",
        help="Usar FP16 (requiere CUDA)"
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Archivo JSON donde guardar resultados (default: moe_results.json)"
    )
    p.add_argument(
        "--batch", action="store_true",
        help="Modo batch: procesar carpeta completa"
    )
    p.add_argument(
        "--knn_k", type=int, default=5,
        help="Número de vecinos para el router k-NN (default: 5)"
    )
    p.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suprimir output de consola detallado"
    )
    p.add_argument(
        "--no_save", action="store_true",
        help="No guardar resultados en JSON"
    )

    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    device = args.device or get_device(prefer_gpu=True)

    # ── Cargar modelo ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Sistema MoE — Inferencia Médica Multimodal")
    logger.info("=" * 60)

    with Timer("Carga del modelo"):
        model = MoEModel(
            base_dir = args.base_dir,
            device   = device,
            use_fp16 = args.fp16,
            knn_k    = args.knn_k,
        )
    model.eval()

    # ── Inferencia ────────────────────────────────────────────────────────
    input_path = Path(args.input)
    verbose    = not args.quiet

    if args.batch or input_path.is_dir():
        if not input_path.is_dir():
            logger.error(f"--batch requiere una carpeta, no un archivo: {input_path}")
            sys.exit(1)
        results = run_batch(model, input_path, verbose=verbose)
    else:
        if not input_path.exists():
            logger.error(f"Archivo no encontrado: {input_path}")
            sys.exit(1)
        results = [run_single(model, input_path, verbose=verbose)]

    # ── Guardar JSON ──────────────────────────────────────────────────────
    if not args.no_save:
        out_path = Path(args.output) if args.output else Path("moe_results.json")
        save_result_json(
            results if len(results) > 1 else results[0],
            out_path,
            append=not args.no_save,
        )

    # ── Resumen batch ─────────────────────────────────────────────────────
    if len(results) > 1:
        successful = sum(1 for r in results if "error" not in r)
        experts_used = {}
        for r in results:
            if "error" not in r:
                eid = r["expert_id"]
                experts_used[eid] = experts_used.get(eid, 0) + 1
        logger.info(f"\n[Resumen] {successful}/{len(results)} exitosos")
        logger.info(f"[Resumen] Expertos usados: {experts_used}")

    return results


if __name__ == "__main__":
    main()
