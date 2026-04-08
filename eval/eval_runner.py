# eval/eval_runner.py
#
# Revised to support:
#   - retrieval evaluation
#   - within-group cosine-distance evaluation in universal space
#   - "base" (no projection; uses raw embeddings)
#   - "base_abttz" (no projection; apply ABTT+zscore to embeddings before eval)
#
# IMPORTANT:
# Put the distance evaluator file somewhere importable, e.g.
#   eval/universal_embedding_distance_eval.py
# so the import below works.

import os
from datetime import datetime
from typing import Callable, Optional, Sequence, Tuple, Dict, Any

import numpy as np
import torch

from eval.cka import linear_cka_from_embeddings
from eval.retreival import UniversalEmbeddingRetrievalEvaluator
from eval.universal_embedding_distance_eval import UniversalEmbeddingDistanceEvaluator

from models.muse import BitextSentenceEncoder
from models.ot import SinkhornOT
from models.dvcca import BitextDVCCA, CachedBitextDVCCA


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
        E = embed_fn(sentences)
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

            if isinstance(Va, dict) or isinstance(Vb, dict):
                continue
            if isinstance(Va, BitextSentenceEncoder) or isinstance(Vb, BitextSentenceEncoder):
                continue
            if isinstance(Va, SinkhornOT) or isinstance(Vb, SinkhornOT):
                continue
            if isinstance(Va, CachedBitextDVCCA) or isinstance(Vb, CachedBitextDVCCA):
                continue

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
# evaluator factories
# -----------------------------
def _make_retrieval_evaluator(
    *,
    V: Optional[np.ndarray],
    embed_fn: Callable[[list[str]], np.ndarray],
    projection_mode: str,
    batch_size: int = 64,
) -> UniversalEmbeddingRetrievalEvaluator:
    return UniversalEmbeddingRetrievalEvaluator(
        V=V,
        embed_fn=embed_fn,
        projection_mode=projection_mode,
        batch_size=batch_size,
    )


def _make_distance_evaluator(
    *,
    V: Optional[np.ndarray],
    embed_fn: Callable[[list[str]], np.ndarray],
    projection_mode: str,
    batch_size: int = 64,
) -> UniversalEmbeddingDistanceEvaluator:
    return UniversalEmbeddingDistanceEvaluator(
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
    name_to_V: Dict[str, Optional[torch.Tensor]],
    embed_fn: Callable[[list[str]], np.ndarray],
    projection_mode: str,
    retrieval_groups,
    retrieval_groups_2,
    retrieval_langs,
    retrieval_K: int = 10,
    retrieval_trials: int = 1000,
    distance_require_different_langs: bool = True,
    distance_return_group_summaries: bool = False,
    seed: int = 0,
    results_dir: str = "results",
    base_abtt_remove: int = 2,
) -> str:
    _ensure_dir(results_dir)
    out_path = os.path.join(results_dir, f"exp_{exp_number}.txt")

    # --- CKA (projection methods only)
    cka_names, cka_mat = _cka_matrix(name_to_V)
    cka_text = _format_cka(cka_names, cka_mat)

    # --- Retrieval
    retrieval_results: Dict[str, Any] = {}

    # --- Distance
    distance_results: Dict[str, Any] = {}

    for name, V_torch in name_to_V.items():
        local_embed_fn = embed_fn
        if name in ("base_abttz", "base+abtt+z", "base_abtt+z", "base_abttz+z"):
            local_embed_fn = _wrap_embed_fn_with_abttz(embed_fn, n_remove=base_abtt_remove)

        if V_torch is None:
            V = None
        elif (
            isinstance(V_torch, dict)
            or isinstance(V_torch, BitextSentenceEncoder)
            or isinstance(V_torch, SinkhornOT)
            or isinstance(V_torch, CachedBitextDVCCA)
        ):
            V = V_torch
        else:
            V = V_torch.detach().cpu().numpy().astype(np.float32)

        retrieval_ev = _make_retrieval_evaluator(
            V=V,
            embed_fn=local_embed_fn,
            projection_mode=projection_mode,
            batch_size=64,
        )
        distance_ev = _make_distance_evaluator(
            V=V,
            embed_fn=local_embed_fn,
            projection_mode=projection_mode,
            batch_size=64,
        )

        # -----------------
        # Retrieval
        # -----------------
        if isinstance(retrieval_ev.V, dict):
            retrieval_report = retrieval_ev.evaluate_2(
                retrieval_groups_2,
                langs=retrieval_langs,
                K=retrieval_K,
                n_trials=retrieval_trials,
                seed=seed,
                hard_negatives=False,
                recall_ks=(1, 3, 5),
                return_details=False,
            )
        elif isinstance(retrieval_ev.V, BitextSentenceEncoder):
            retrieval_report = retrieval_ev.evaluate_3(
                retrieval_groups,
                langs=retrieval_langs,
                encoder=retrieval_ev.V,
                n_trials=retrieval_trials,
                K=retrieval_K,
                recall_ks=(1, 3, 5),
                seed=seed,
            )
        elif isinstance(V, SinkhornOT):
            retrieval_report = retrieval_ev.evaluate_4(
                retrieval_groups,
                langs=retrieval_langs,
                ot_model=retrieval_ev.V,
                n_trials=retrieval_trials,
                K=retrieval_K,
                recall_ks=(1, 3, 5),
                seed=seed,
                hard_negatives=False,
            )
        elif isinstance(V, CachedBitextDVCCA):
            retrieval_report = retrieval_ev.evaluate_5(
                retrieval_groups,
                dvcca_model=retrieval_ev.V,
                langs=retrieval_langs,
                n_trials=retrieval_trials,
                K=retrieval_K,
                recall_ks=(1, 3, 5),
                seed=seed,
            )
        else:
            retrieval_report = retrieval_ev.evaluate(
                retrieval_groups,
                langs=retrieval_langs,
                K=retrieval_K,
                n_trials=retrieval_trials,
                seed=seed,
                hard_negatives=False,
                recall_ks=(1, 3, 5),
                return_details=False,
            )

        retrieval_results[name] = retrieval_report

        # -----------------
        # Distance
        # -----------------
        if isinstance(distance_ev.V, dict):
            distance_report = distance_ev.evaluate_2(
                retrieval_groups_2,
                langs=retrieval_langs,
                require_different_langs=distance_require_different_langs,
                return_details=False,
                return_group_summaries=distance_return_group_summaries,
            )
        elif isinstance(distance_ev.V, BitextSentenceEncoder):
            distance_report = distance_ev.evaluate_3(
                retrieval_groups,
                langs=retrieval_langs,
                encoder=distance_ev.V,
                require_different_langs=distance_require_different_langs,
                return_details=False,
                return_group_summaries=distance_return_group_summaries,
            )
        elif isinstance(V, SinkhornOT):
            distance_report = distance_ev.evaluate_4(
                retrieval_groups,
                langs=retrieval_langs,
                ot_model=distance_ev.V,
                require_different_langs=distance_require_different_langs,
                return_details=False,
                return_group_summaries=distance_return_group_summaries,
            )
        elif isinstance(V, CachedBitextDVCCA):
            distance_report = distance_ev.evaluate_5(
                retrieval_groups,
                dvcca_model=distance_ev.V,
                langs=retrieval_langs,
                require_different_langs=distance_require_different_langs,
                return_details=False,
                return_group_summaries=distance_return_group_summaries,
            )
        else:
            distance_report = distance_ev.evaluate(
                retrieval_groups,
                langs=retrieval_langs,
                require_different_langs=distance_require_different_langs,
                return_details=False,
                return_group_summaries=distance_return_group_summaries,
            )

        distance_results[name] = distance_report

    # --- Write report
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: exp_{exp_number}\n")
        f.write(f"Generated:  {now}\n\n")

        f.write("=== SETTINGS ===\n")
        f.write(f"projection_mode                 = {projection_mode}\n")
        f.write(f"retrieval_K                    = {retrieval_K}\n")
        f.write(f"retrieval_trials               = {retrieval_trials}\n")
        f.write(f"distance_require_different_langs = {distance_require_different_langs}\n")
        f.write(f"distance_return_group_summaries  = {distance_return_group_summaries}\n")
        f.write(f"seed                           = {seed}\n")
        f.write(f"base_abtt_remove               = {base_abtt_remove}\n\n")

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

        f.write("\n=== DISTANCE RESULTS ===\n")
        for name in sorted(distance_results.keys()):
            r = distance_results[name]
            f.write(f"\n[{name}]\n")
            f.write(f"Avg cosine similarity: {r.avg_cosine_similarity:.6f}\n")
            f.write(f"Avg cosine distance:   {r.avg_cosine_distance:.6f}\n")
            f.write(f"Std cosine distance:   {r.std_cosine_distance:.6f}\n")
            f.write(f"Min cosine distance:   {r.min_cosine_distance:.6f}\n")
            f.write(f"Max cosine distance:   {r.max_cosine_distance:.6f}\n")
            f.write(f"Groups used:           {r.n_groups}\n")
            f.write(f"Pairs used:            {r.n_pairs}\n")

            if r.group_summaries is not None:
                f.write("Group summaries:\n")
                for gs in r.group_summaries:
                    f.write(
                        f"  group={gs.group_id} n_sentences={gs.n_sentences} "
                        f"n_pairs={gs.n_pairs} avg_sim={gs.avg_cosine_similarity:.6f} "
                        f"avg_dist={gs.avg_cosine_distance:.6f}\n"
                    )

    return out_path
