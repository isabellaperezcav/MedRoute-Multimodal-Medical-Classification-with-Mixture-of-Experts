"""
router_knn.py — Router k-NN con FAISS (cosine similarity)
Router ganador del ablation study: Routing Accuracy = 1.0
IndexFlatIP sobre embeddings L2-normalizados → cosine similarity exacta.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Union


class KNNRouter:
    """
    Router basado en k-NN con FAISS (cosine similarity via IndexFlatIP).

    El routing depende ÚNICAMENTE del embedding CLS; no se usa ningún
    metadato externo ni label de dataset.

    Parámetros
    ----------
    embeddings_path : ruta a all_train_Z.npy      (N, d_model)
    labels_path     : ruta a all_train_expert_y.npy (N,) con expert_id 0-4
    k               : vecinos a consultar (default=5)
    device          : "cpu" o "cuda" (FAISS-GPU si disponible)
    """

    def __init__(
        self,
        embeddings_path: Union[str, Path],
        labels_path:     Union[str, Path],
        k:               int  = 5,
        device:          str  = "cpu",
    ):
        import faiss

        self.k      = k
        self.device = device

        # ── Cargar embeddings de entrenamiento ────────────────────────────
        Z = np.load(str(embeddings_path)).astype(np.float32)  # (N, d)
        y = np.load(str(labels_path)).astype(np.int32)        # (N,)

        assert Z.shape[0] == y.shape[0], \
            f"Mismatch: {Z.shape[0]} embeddings vs {y.shape[0]} labels"

        self.labels = y
        self.d      = Z.shape[1]

        # ── Normalizar L2 → cosine similarity ────────────────────────────
        faiss.normalize_L2(Z)

        # ── Construir índice ─────────────────────────────────────────────
        self.index = faiss.IndexFlatIP(self.d)   # Inner Product = cosine tras norm

        # Intentar GPU FAISS si device es cuda
        self._use_gpu = False
        if "cuda" in device:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                self._use_gpu = True
                print("[Router] FAISS en GPU")
            except Exception:
                print("[Router] FAISS GPU no disponible, usando CPU")

        self.index.add(Z)
        print(f"[Router] k-NN FAISS inicializado | "
              f"N={Z.shape[0]} | d={self.d} | k={self.k} | "
              f"expertos únicos={np.unique(y).tolist()}")

    # ── Predicción (tensor individual) ───────────────────────────────────

    def predict(self, z: Union[np.ndarray, torch.Tensor]) -> int:
        """
        Recibe embedding CLS de UNA muestra → devuelve expert_id (0-4).

        z : (d_model,) o (1, d_model) — numpy o torch
        """
        import faiss
        z_np = self._to_numpy(z)                   # (1, d)
        faiss.normalize_L2(z_np)
        _, I = self.index.search(z_np, self.k)     # I: (1, k)
        expert_ids = self.labels[I[0]]             # (k,)
        # Votación por mayoría
        counts = np.bincount(expert_ids.astype(int),
                             minlength=int(self.labels.max()) + 1)
        return int(counts.argmax())

    # ── Predicción en batch ───────────────────────────────────────────────

    def predict_batch(
        self,
        Z: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Recibe batch de embeddings → devuelve array de expert_ids.

        Z : (B, d_model)
        Retorna: (B,) int32
        """
        import faiss
        Z_np = self._to_numpy(Z)                   # (B, d)
        faiss.normalize_L2(Z_np)
        _, I = self.index.search(Z_np, self.k)     # (B, k)
        B    = Z_np.shape[0]
        n_experts = int(self.labels.max()) + 1
        preds = np.empty(B, dtype=np.int32)
        for i in range(B):
            ids    = self.labels[I[i]]
            counts = np.bincount(ids.astype(int), minlength=n_experts)
            preds[i] = counts.argmax()
        return preds

    # ── Top-k vecinos con scores (para logging) ───────────────────────────

    def predict_with_scores(
        self,
        z: Union[np.ndarray, torch.Tensor],
    ) -> dict:
        """
        Devuelve dict con expert_id, scores de vecinos y distribución de votos.
        Útil para el log de decisiones del router.
        """
        import faiss
        z_np = self._to_numpy(z)
        faiss.normalize_L2(z_np)
        D, I = self.index.search(z_np, self.k)
        expert_ids  = self.labels[I[0]]
        scores      = D[0].tolist()
        n_experts   = int(self.labels.max()) + 1
        counts      = np.bincount(expert_ids.astype(int), minlength=n_experts)
        expert_id   = int(counts.argmax())
        votes       = {int(e): int(c) for e, c in enumerate(counts) if c > 0}

        return {
            "expert_id":      expert_id,
            "neighbor_ids":   I[0].tolist(),
            "neighbor_labels":expert_ids.tolist(),
            "cosine_scores":  scores,
            "vote_counts":    votes,
            "confidence":     float(max(counts) / self.k),
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _to_numpy(z: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()
        z = z.astype(np.float32)
        if z.ndim == 1:
            z = z[np.newaxis, :]     # (1, d)
        return np.ascontiguousarray(z)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_router(
    embeddings_dir: Union[str, Path] = "embedings/Output embeddings",
    k:    int = 5,
    device: str = "cpu",
) -> KNNRouter:
    base = Path(embeddings_dir)
    return KNNRouter(
        embeddings_path = base / "all_train_Z.npy",
        labels_path     = base / "all_train_expert_y.npy",
        k               = k,
        device          = device,
    )
