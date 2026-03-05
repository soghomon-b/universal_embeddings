# eval/eval_runner.py
#
# Revised to support:
#   - "base" (no projection; uses raw embeddings)
#   - "base_abttz" (no projection; apply ABTT+zscore to embeddings before retrieval)
# alongside your existing methods that provide projection matrices V (shape [d,k]).
#
# Output format:
# === RETRIEVAL RESULTS ===
# [base] ...
# [base_abttz] ...
# [geometric] ...
# [infonce] ...
# [pairwise] ...
# [supcon] ...
#
# Notes:
# - CKA is only computed between methods that actually have a V (base/base_abttz excluded).
# - ABTT+zscore is applied per-evaluation-call on the embedded sentences (not on training TSV).
# - This assumes your UniversalEmbeddingRetrievalEvaluator can handle V=None (no projection).
#   If it can't, see the fallback comment in _make_evaluator() below.

import os
from datetime import datetime
from typing import Callable, Optional, Sequence, Tuple, Dict, Any

import numpy as np
import torch

from eval.cka import linear_cka_from_embeddings
from eval.retreival import UniversalEmbeddingRetrievalEvaluator


# -----------------------------
# utils
# -----------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_torch_2d(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.tensor(x)
    if t.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {tuple(t.shape)}")
    return t


# -----------------------------
# ABTT + z-score (torch), applied to a batch of embeddings
# -----------------------------
def abtt_and_zscore_torch(X: torch.Tensor, n_remove: int = 2) -> torch.Tensor:
    """
    X: (N, d) float32
    - global center
    - PCA lowrank
    - remove top n_remove PCs (ABTT)
    - global z-score
    """
    X = X.float()
    X_centered = X - X.mean(dim=0, keepdim=True)

    n_samples, d = X_centered.shape
    q = min(max(n_remove + 2, n_remove), min(n_samples, d))
    # torch.pca_lowrank expects (N,d)
    _, _, V = torch.pca_lowrank(X_centered, q=q)  # V: (d, q)

    comps = V[:, :n_remove].T  # (n_remove, d)
    proj = X_centered @ comps.T
    X_debiased = X_centered - proj @ comps

    mean2 = X_debiased.mean(dim=0, keepdim=True)
    std2 = X_debiased.std(dim=0, keepdim=True, unbiased=False) + 1e-8
    return (X_debiased - mean2) / std2


def _wrap_embed_fn_with_abttz(
    embed_fn: Callable[[list[str]], np.ndarray],
    n_remove: int = 2,
) -> Callable[[list[str]], np.ndarray]:
    """
    Returns a new embed_fn that applies ABTT+zscore after embedding.
    Uses torch for PCA, returns numpy float32.
    """
    def wrapped(sentences: list[str]) -> np.ndarray:
        E = embed_fn(sentences)  # (n,d) np
        Et = torch.tensor(E, dtype=torch.float32)
        Et2 = abtt_and_zscore_torch(Et, n_remove=n_remove)
        return Et2.cpu().numpy().astype(np.float32)

    return wrapped


# -----------------------------
# CKA between projection matrices (exclude base)
# -----------------------------
def _cka_matrix(name_to_V: Dict[str, Optional[torch.Tensor]]) -> Tuple[list[str], torch.Tensor]:
    """
    CKA between projection matrices only.
    base/base_abttz are excluded (V is None).
    """
    names = [n for n, V in name_to_V.items() if V is not None]
    n = len(names)
    M = torch.zeros((n, n), dtype=torch.float32)

    for i in range(n):
        for j in range(n):
            Va = name_to_V[names[i]]
            Vb = name_to_V[names[j]]
            assert Va is not None and Vb is not None

            Va2 = _to_torch_2d(Va)
            Vb2 = _to_torch_2d(Vb)

            if Va2.shape[0] != Vb2.shape[0]:
                raise ValueError(
                    f"CKA needs same #rows: {names[i]} {Va2.shape} vs {names[j]} {Vb2.shape}"
                )

            M[i, j] = linear_cka_from_embeddings(Va2, Vb2).float()

    return names, M


def _format_cka(names: list[str], M: torch.Tensor) -> str:
    if len(names) == 0:
        return "(no projection matrices to compare)"
    colw = max(10, max(len(n) for n in names) + 2)
    header = " " * colw + "".join(n.rjust(colw) for n in names)
    lines = [header]
    for i, rowname in enumerate(names):
        row = rowname.ljust(colw) + "".join(
            f"{M[i,j].item():.4f}".rjust(colw) for j in range(len(names))
        )
        lines.append(row)
    return "\n".join(lines)


# -----------------------------
# evaluator factory (supports base via V=None)
# -----------------------------
def _make_evaluator(
    *,
    V: Optional[np.ndarray],
    embed_fn: Callable[[list[str]], np.ndarray],
    projection_mode: str,
    batch_size: int = 64,
) -> UniversalEmbeddingRetrievalEvaluator:
    """
    Requires UniversalEmbeddingRetrievalEvaluator to accept V=None as 'no projection'.
    If your evaluator DOES NOT support that, you have two options:
      (1) modify the evaluator to treat V=None as identity/no-projection (recommended), OR
      (2) pass an identity matrix as V and ensure projection_mode behaves like identity.
    """
    return UniversalEmbeddingRetrievalEvaluator(
        V=V,
        embed_fn=embed_fn,
        projection_mode=projection_mode,
        batch_size=batch_size,
    )


# -----------------------------
# main entry point
# -----------------------------
def run_full_eval(
    *,
    exp_number: int,
    name_to_V: Dict[str, Optional[torch.Tensor]],  # V is [d,k] torch on CPU, or None for base
    embed_fn: Callable[[list[str]], np.ndarray],   # callable(list[str]) -> np.ndarray [n,d]
    projection_mode: str,
    retrieval_groups,  # output of extract_parallel_maxcover
    retrieval_langs,   # same shape as groups, or None
    retrieval_K: int = 10,
    retrieval_trials: int = 1000,
    seed: int = 0,
    results_dir: str = "results",
    base_abtt_remove: int = 2,  # ABTT remove count used for base_abttz
) -> str:
    _ensure_dir(results_dir)
    out_path = os.path.join(results_dir, f"exp_{exp_number}.txt")

    # --- CKA (projection methods only)
    cka_names, cka_mat = _cka_matrix(name_to_V)
    cka_text = _format_cka(cka_names, cka_mat)

    # --- Retrieval
    retrieval_results: Dict[str, Any] = {}

    for name, V_torch in name_to_V.items():
        # pick embed_fn variant
        local_embed_fn = embed_fn
        if name in ("base_abttz", "base+abtt+z", "base_abtt+z", "base_abttz+z"):
            local_embed_fn = _wrap_embed_fn_with_abttz(embed_fn, n_remove=base_abtt_remove)

        # convert V
        if V_torch is None:
            V = None
        else:
            V = V_torch.detach().cpu().numpy().astype(np.float32)

        ev = _make_evaluator(
            V=V,
            embed_fn=local_embed_fn,
            projection_mode=projection_mode,
            batch_size=64,
        )

        report = ev.evaluate(
            retrieval_groups,
            langs=retrieval_langs,
            K=retrieval_K,
            n_trials=retrieval_trials,
            seed=seed,
            hard_negatives=False,
            recall_ks=(1, 3, 5),
            return_details=False,
        )

        retrieval_results[name] = report

    # --- Write report
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: exp_{exp_number}\n")
        f.write(f"Generated:  {now}\n\n")

        f.write("=== SETTINGS ===\n")
        f.write(f"projection_mode = {projection_mode}\n")
        f.write(f"retrieval_K      = {retrieval_K}\n")
        f.write(f"retrieval_trials = {retrieval_trials}\n")
        f.write(f"seed             = {seed}\n")
        f.write(f"base_abtt_remove = {base_abtt_remove}\n\n")

        f.write("=== CKA (between projection matrices V) ===\n")
        f.write(cka_text + "\n\n")

        f.write("=== RETRIEVAL RESULTS ===\n")
        for name in sorted(retrieval_results.keys()):
            r = retrieval_results[name]
            f.write(f"\n[{name}]\n")
            f.write(f"Accuracy@1: {r.accuracy_at_1:.4f}\n")
            f.write(f"MRR:        {r.mrr:.4f}\n")
            f.write(
                "Recall@k:   "
                + ", ".join(f"{k}:{v:.4f}" for k, v in r.recall_at_k.items())
                + "\n"
            )

    return out_path