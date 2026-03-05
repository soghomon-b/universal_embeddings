import os
import itertools
from datetime import datetime

import numpy as np
import torch

from eval.cka import linear_cka_from_embeddings
from eval.retreival import UniversalEmbeddingRetrievalEvaluator


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


def _cka_matrix(name_to_V: dict[str, torch.Tensor]) -> tuple[list[str], torch.Tensor]:
    """
    CKA between projection matrices.
    Interpreting each V as a set of k vectors in R^d (or d vectors in R^k) is ambiguous,
    but we can compare consistently by flattening to a common orientation.
    Here we compare as [d,k] matrices directly by treating rows as "samples".
    """
    names = list(name_to_V.keys())
    n = len(names)
    M = torch.zeros((n, n), dtype=torch.float32)

    for i in range(n):
        for j in range(n):
            Va = name_to_V[names[i]]
            Vb = name_to_V[names[j]]

            # CKA expects: Va: [n, da], Vb: [n, db]
            # We'll use n=d (rows), da=k. This compares column-subspaces in a rough sense.
            Va2 = _to_torch_2d(Va)
            Vb2 = _to_torch_2d(Vb)

            # Ensure same "n" dimension
            if Va2.shape[0] != Vb2.shape[0]:
                raise ValueError(
                    f"CKA needs same #rows: {names[i]} {Va2.shape} vs {names[j]} {Vb2.shape}"
                )

            M[i, j] = linear_cka_from_embeddings(Va2, Vb2).float()

    return names, M


def _format_cka(names: list[str], M: torch.Tensor) -> str:
    # simple aligned table
    colw = max(10, max(len(n) for n in names) + 2)
    header = " " * colw + "".join(n.rjust(colw) for n in names)
    lines = [header]
    for i, rowname in enumerate(names):
        row = rowname.ljust(colw) + "".join(
            f"{M[i,j].item():.4f}".rjust(colw) for j in range(len(names))
        )
        lines.append(row)
    return "\n".join(lines)


def run_full_eval(
    *,
    exp_number: int,
    name_to_V: dict[str, torch.Tensor],  # each V is [d,k] torch tensor on CPU
    embed_fn,  # callable(list[str]) -> np.ndarray [n,d]
    projection_mode: str,
    retrieval_groups,  # output of extract_parallel_maxcover
    retrieval_langs,  # same shape as groups, or None
    retrieval_K: int = 10,
    retrieval_trials: int = 1000,
    seed: int = 0,
    results_dir: str = "results",
) -> str:
    _ensure_dir(results_dir)
    out_path = os.path.join(results_dir, f"exp_{exp_number}.txt")

    # --- CKA
    cka_names, cka_mat = _cka_matrix(name_to_V)
    cka_text = _format_cka(cka_names, cka_mat)

    # --- Retrieval
    retrieval_results = {}
    for name, V_torch in name_to_V.items():
        V = V_torch.detach().cpu().numpy().astype(np.float32)

        ev = UniversalEmbeddingRetrievalEvaluator(
            V=V,
            embed_fn=embed_fn,
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
        f.write(f"seed             = {seed}\n\n")

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
