import os
from collections import defaultdict, Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data_loader import (
    SplitConfig,
    prepare_parallel_data,
    HFEmbedder,
    DiskEmbeddingCache,
    CachedEmbedder,
)


# ============================================================
# 1) Embedding helper
# ============================================================

@torch.no_grad()
def embed_sentences_in_batches(
    sentences: List[str],
    embed_fn: Callable[[Sequence[str]], torch.Tensor],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    if len(sentences) == 0:
        raise ValueError("embed_sentences_in_batches got an empty sentence list")

    embs = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i:i + batch_size]
        e = embed_fn(chunk)

        if not torch.is_tensor(e):
            e = torch.tensor(e, dtype=torch.float32)

        if e.ndim == 1:
            e = e.unsqueeze(0)

        e = e.float().to(device)
        embs.append(e)

    return torch.cat(embs, dim=0)


# ============================================================
# 2) Collect per-language views
# ============================================================

@torch.no_grad()
def collect_language_views_from_loader(
    loader: DataLoader,
    embed_fn_by_lang: Dict[str, Callable[[Sequence[str]], torch.Tensor]],
    device: str = "cpu",
    embed_batch_size: int = 64,
):
    """
    Returns:
        views[lang] = (X_lang, idx_lang)
        n_total = number of shared item ids
        lang_counts = count of examples per language
    """
    emb_lists = defaultdict(list)
    idx_lists = defaultdict(list)
    lang_counts = Counter()

    global_row = 0

    for src_langs, tgt_langs, s1, s2 in loader:
        B = len(s1)

        src_group = defaultdict(list)
        src_pos = defaultdict(list)
        for i, (lang, sent) in enumerate(zip(src_langs, s1)):
            src_group[lang].append(sent)
            src_pos[lang].append(global_row + i)
            lang_counts[lang] += 1

        tgt_group = defaultdict(list)
        tgt_pos = defaultdict(list)
        for i, (lang, sent) in enumerate(zip(tgt_langs, s2)):
            tgt_group[lang].append(sent)
            tgt_pos[lang].append(global_row + i)
            lang_counts[lang] += 1

        for lang, sents in src_group.items():
            if lang not in embed_fn_by_lang:
                raise KeyError(f"No embedder found for source language '{lang}'")
            E = embed_sentences_in_batches(
                sents, embed_fn_by_lang[lang], embed_batch_size, device
            )
            emb_lists[lang].append(E)
            idx_lists[lang].append(
                torch.tensor(src_pos[lang], device=device, dtype=torch.long)
            )

        for lang, sents in tgt_group.items():
            if lang not in embed_fn_by_lang:
                raise KeyError(f"No embedder found for target language '{lang}'")
            E = embed_sentences_in_batches(
                sents, embed_fn_by_lang[lang], embed_batch_size, device
            )
            emb_lists[lang].append(E)
            idx_lists[lang].append(
                torch.tensor(tgt_pos[lang], device=device, dtype=torch.long)
            )

        global_row += B

    views = {}
    for lang in emb_lists:
        X_lang = torch.cat(emb_lists[lang], dim=0)
        idx_lang = torch.cat(idx_lists[lang], dim=0)
        views[lang] = (X_lang, idx_lang)

    return views, global_row, lang_counts


# ============================================================
# 3) VecMap-style multilingual model
# ============================================================

class VecMapProjector(nn.Module):
    """
    Maps every language into a chosen hub language space.
    Hub language uses identity.
    Non-hub languages use orthogonal linear maps.
    """

    def __init__(self, dims_by_lang: Dict[str, int], hub_lang: str):
        super().__init__()
        self.langs = sorted(dims_by_lang.keys())
        self.hub_lang = hub_lang
        self.dims_by_lang = dims_by_lang

        if hub_lang not in dims_by_lang:
            raise KeyError(f"hub_lang '{hub_lang}' not found in dims_by_lang")

        hub_dim = dims_by_lang[hub_lang]
        self.projs = nn.ModuleDict()

        for lang, d in dims_by_lang.items():
            if d != hub_dim:
                raise ValueError(
                    f"All dims must match for this simplified VecMap. "
                    f"Got {lang}: {d}, hub {hub_lang}: {hub_dim}"
                )

            layer = nn.Linear(d, hub_dim, bias=False)
            layer.weight.data.copy_(torch.eye(hub_dim, dtype=layer.weight.dtype))
            self.projs[lang] = layer

        self.means_by_lang: Dict[str, torch.Tensor] = {}

    def forward(self, lang: str, x: torch.Tensor) -> torch.Tensor:
        if lang not in self.projs:
            raise KeyError(f"Unknown language: {lang}")

        mean = self.means_by_lang.get(lang, None)
        if mean is not None:
            x = x - mean.to(device=x.device, dtype=x.dtype)

        z = self.projs[lang](x)
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def project_language_matrix(self, lang: str, X: torch.Tensor) -> torch.Tensor:
        return self.forward(lang, X)


# ============================================================
# 4) VecMap utilities
# ============================================================

@torch.no_grad()
def length_normalize(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return X / X.norm(dim=1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def center_and_normalize(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    Xc = length_normalize(Xc)
    return Xc, mean


@torch.no_grad()
def orthogonal_procrustes(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Solve min_W ||XW - Y|| subject to W^T W = I
    X: (n, d), Y: (n, d)
    Returns W: (d, d)
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"Expected 2D matrices, got X={tuple(X.shape)} Y={tuple(Y.shape)}")

    if X.shape != Y.shape:
        raise ValueError(f"Shape mismatch for Procrustes: X={tuple(X.shape)} Y={tuple(Y.shape)}")

    M = X.T @ Y
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    W = U @ Vt
    return W


@torch.no_grad()
def intersect_by_item_ids(
    idx_a: torch.Tensor,
    idx_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns positions into A and B corresponding to shared item ids.
    """
    pos_b = {int(v): j for j, v in enumerate(idx_b.tolist())}
    a_pos = []
    b_pos = []

    for i, v in enumerate(idx_a.tolist()):
        j = pos_b.get(int(v), None)
        if j is not None:
            a_pos.append(i)
            b_pos.append(j)

    if not a_pos:
        return (
            torch.empty(0, dtype=torch.long, device=idx_a.device),
            torch.empty(0, dtype=torch.long, device=idx_b.device),
        )

    return (
        torch.tensor(a_pos, dtype=torch.long, device=idx_a.device),
        torch.tensor(b_pos, dtype=torch.long, device=idx_b.device),
    )


@torch.no_grad()
def nearest_neighbor_dictionary(
    X_src_mapped: torch.Tensor,
    X_tgt: torch.Tensor,
    max_pairs: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each row in X_src_mapped, find nearest neighbor in X_tgt by cosine.
    Returns matched row indices.
    """
    X_src_mapped = length_normalize(X_src_mapped)
    X_tgt = length_normalize(X_tgt)

    sims = X_src_mapped @ X_tgt.T
    nn_idx = sims.argmax(dim=1)

    src_idx = torch.arange(X_src_mapped.shape[0], device=X_src_mapped.device)
    tgt_idx = nn_idx

    if max_pairs is not None and src_idx.numel() > max_pairs:
        keep = torch.randperm(src_idx.numel(), device=src_idx.device)[:max_pairs]
        src_idx = src_idx[keep]
        tgt_idx = tgt_idx[keep]

    return src_idx, tgt_idx


# ============================================================
# 5) Fit multilingual VecMap-style aligner
# ============================================================

@torch.no_grad()
def fit_vecmap_multilingual(
    views: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    hub_lang: str,
    num_iters: int = 5,
    max_pairs_per_iter: Optional[int] = 20000,
    verbose: bool = True,
) -> VecMapProjector:
    """
    Learns one map per language into the hub language space.

    views[lang] = (X_lang, idx_lang)
      X_lang:   (n_lang, d)
      idx_lang: (n_lang,) shared item ids
    """
    if hub_lang not in views:
        raise KeyError(f"hub_lang '{hub_lang}' not found in views")

    device = next(iter(views.values()))[0].device
    dtype = next(iter(views.values()))[0].dtype

    dims_by_lang = {lang: X.shape[1] for lang, (X, _) in views.items()}
    hub_dim = dims_by_lang[hub_lang]

    for lang, d in dims_by_lang.items():
        if d != hub_dim:
            raise ValueError(
                f"All language views must have same dim for simplified VecMap. "
                f"Got {lang}: {d}, hub {hub_lang}: {hub_dim}"
            )

    model = VecMapProjector(dims_by_lang=dims_by_lang, hub_lang=hub_lang).to(device)

    proc_views = {}
    for lang, (X, idx) in views.items():
        X = X.to(device=device, dtype=dtype)
        idx = idx.to(device=device, dtype=torch.long)

        Xn, mean = center_and_normalize(X)
        proc_views[lang] = (Xn, idx)
        model.means_by_lang[lang] = mean.detach().cpu()

    X_hub, idx_hub = proc_views[hub_lang]
    d = X_hub.shape[1]

    # hub is exact identity in true hub dim
    model.projs[hub_lang].weight.copy_(torch.eye(d, device=device, dtype=dtype))

    for lang in model.langs:
        if lang == hub_lang or lang not in proc_views:
            continue

        X_lang, idx_lang = proc_views[lang]

        src_pos, hub_pos = intersect_by_item_ids(idx_lang, idx_hub)

        if src_pos.numel() == 0:
            if verbose:
                print(f"[VecMap] {lang}: no true overlap with hub '{hub_lang}', using identity init")
            model.projs[lang].weight.copy_(torch.eye(d, device=device, dtype=dtype))
            continue

        W = orthogonal_procrustes(X_lang[src_pos], X_hub[hub_pos])
        model.projs[lang].weight.copy_(W.T)

        if verbose:
            print(f"[VecMap] {lang}: init pairs = {src_pos.numel()}")

        for it in range(num_iters):
            X_lang_mapped = length_normalize(X_lang @ W)

            src_idx, tgt_idx = nearest_neighbor_dictionary(
                X_lang_mapped,
                X_hub,
                max_pairs=max_pairs_per_iter,
            )

            W = orthogonal_procrustes(X_lang[src_idx], X_hub[tgt_idx])
            model.projs[lang].weight.copy_(W.T)

            if verbose:
                print(f"[VecMap] {lang}: iter {it + 1}/{num_iters}, pseudo-pairs = {src_idx.numel()}")

    return model


# ============================================================
# 6) End-to-end runner
# ============================================================

def run_vecmap_training_example(
    tsv_path: str,
    seed: int,
    subset_size: int = 50000,
    batch_size_pairs: int = 256,
    embed_batch_size: int = 64,
    hub_lang: Optional[str] = None,
    num_iters: int = 5,
    max_pairs_per_iter: Optional[int] = 20000,
    ollama_model: str = "llama3.1:8b",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    cache_dir: str = "./vecmap_cache",
):
    cfg = SplitConfig(
        subset_size=subset_size,
        train_frac=0.90,
        val_frac=0.05,
        test_frac=0.05,
        seed=seed,
    )

    train_loader, val_loader, test_loader, languages = prepare_parallel_data(
        tsv_path=tsv_path,
        cfg=cfg,
        batch_size_pairs=batch_size_pairs,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )
    del val_loader, test_loader

    base = HFEmbedder(model_name=ollama_model, device=device)
    cache = DiskEmbeddingCache(cache_dir)

    embed_fn_by_lang = {
        lang: CachedEmbedder(base, cache)
        for lang in languages
    }

    views, n_total, lang_counts = collect_language_views_from_loader(
        loader=train_loader,
        embed_fn_by_lang=embed_fn_by_lang,
        device=device,
        embed_batch_size=embed_batch_size,
    )
    del n_total

    if not views:
        raise ValueError("No language views were collected")

    if hub_lang is None:
        hub_lang = lang_counts.most_common(1)[0][0]

    print(f"Languages: {languages}")
    print(f"Hub language: {hub_lang}")
    print(f"Detected embedding dim: {next(iter(views.values()))[0].shape[1]}")

    model = fit_vecmap_multilingual(
        views=views,
        hub_lang=hub_lang,
        num_iters=num_iters,
        max_pairs_per_iter=max_pairs_per_iter,
        verbose=True,
    )

    return model


if __name__ == "__main__":
    TSV_PATH = "your_parallel_data.tsv"

    # Smoke test:
    # model = run_vecmap_training_example(
    #     tsv_path=TSV_PATH,
    #     seed=42,
    #     subset_size=500,
    #     batch_size_pairs=64,
    #     embed_batch_size=16,
    #     hub_lang=None,
    #     num_iters=2,
    #     ollama_model="sentence-transformers/all-mpnet-base-v2",
    #     cache_dir="./vecmap_cache",
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    # )

    pass