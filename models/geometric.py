"""
geometric_universal_subspace_from_tsv.py

Goal
----
Run your *geometric* universal-subspace meta-experiment (SVD/merge loop),
but source the data from a parallel TSV (src_lang \t tgt_lang \t src_sent \t tgt_sent).

Key requirements you asked for
------------------------------
1) Use a `run_training_example(...)` entry point (like your PyTorch script).
2) Select a random subset using a seed (same seed as your PyTorch experiments).
3) "Not pairs, just sentences":
   - sample a subset of PAIRS (deterministically via reservoir sampling with seed)
   - then expand into SENTENCES (src + tgt), shuffle deterministically, take subset
4) Embed sentences using Ollama embedding model (granite-embedding:278m) with disk cache.
5) Build datasets X by sampling rows from this sentence embedding pool.
6) Run your meta-loop (K datasets per run; NUM_RUNS meta-runs) and output V_acc.

Notes
-----
- This uses torch for GPU SVD/PCA math (same as your original script),
  but it does NOT train a neural net.
- Determinism:
  - The subset is deterministic given seed (reservoir sampling + deterministic shuffle).
  - The meta-loop dataset construction is deterministic given seed and run offsets.
  - torch SVD/PCA on GPU can still have small nondeterminism; if you need more,
    set `torch.use_deterministic_algorithms(True)` and run on CPU (slower).
"""

from __future__ import annotations

import os
import json
import random
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import requests
import torch


# ============================================================
# 0) Seed helpers (for reproducibility)
# ============================================================


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1) TSV sampling like your PyTorch pipeline (reservoir)
# ============================================================


@dataclass
class SplitConfig:
    subset_size: Optional[int] = None  # number of pairs to sample; None => use all
    seed: int = 42


def parse_parallel_tsv_line(line: str) -> Optional[Tuple[str, str, str, str]]:
    line = line.rstrip("\n")
    if not line:
        return None
    parts = line.split("\t")
    if len(parts) < 4:
        return None
    src_lang, tgt_lang = parts[0], parts[1]
    src_sent = parts[2].strip()
    tgt_sent = parts[3].strip()
    if not src_sent or not tgt_sent:
        return None
    return src_lang, tgt_lang, src_sent, tgt_sent


def reservoir_sample_tsv_pairs(
    path: str, k: int, seed: int
) -> List[Tuple[str, str, str, str]]:
    """
    Deterministic reservoir sample of k *pairs* from TSV.
    This matches the idea in your PyTorch pipeline.
    """
    rng = random.Random(seed)
    sample: List[Tuple[str, str, str, str]] = []
    seen = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = parse_parallel_tsv_line(line)
            if item is None:
                continue

            seen += 1
            if len(sample) < k:
                sample.append(item)
            else:
                j = rng.randrange(seen)  # 0..seen-1
                if j < k:
                    sample[j] = item

    if not sample:
        raise ValueError("No valid pairs parsed from TSV.")
    return sample


def load_all_tsv_pairs(path: str) -> List[Tuple[str, str, str, str]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = parse_parallel_tsv_line(line)
            if item is not None:
                out.append(item)
    if not out:
        raise ValueError("No valid pairs parsed from TSV.")
    return out


from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple, Set


def pairs_to_sentence_pool_with_langs(
    pairs: Sequence[Tuple[str, str, str, str]],
    sentence_subset_size: Optional[int],
    seed: int,
    dedup: bool = True,
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Expand pairs -> sentences (src + tgt). Track lang for each sentence.
    Then deterministic shuffle and take subset.

    Returns:
      sentence_pool: List[str]
      langs_in_pool: sorted list of language codes included (after subset/dedup)
      lang_counts: dict lang -> number of sentences in pool (after subset/dedup)
    """
    sents: List[str] = []
    sent_langs: List[str] = []  # parallel to sents

    for src_lang, tgt_lang, s1, s2 in pairs:
        sents.append(s1)
        sent_langs.append(src_lang)
        sents.append(s2)
        sent_langs.append(tgt_lang)

    if dedup:
        # deterministic dedup while preserving first occurrence
        seen: Set[str] = set()
        s2: List[str] = []
        l2: List[str] = []
        for s, lang in zip(sents, sent_langs):
            if s not in seen:
                seen.add(s)
                s2.append(s)
                l2.append(lang)
        sents, sent_langs = s2, l2

    rng = random.Random(seed)
    idx = list(range(len(sents)))
    rng.shuffle(idx)

    if sentence_subset_size is not None:
        idx = idx[:sentence_subset_size]

    sents = [sents[i] for i in idx]
    sent_langs = [sent_langs[i] for i in idx]

    if not sents:
        raise ValueError("Sentence pool is empty after sampling/dedup/subset.")

    counts = Counter(sent_langs)
    langs_in_pool = sorted(counts.keys())
    return sents, langs_in_pool, dict(counts)


# ============================================================
# 2) Ollama embeddings with disk cache (same as before)
# ============================================================


class DiskEmbeddingCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _path(self, text: str) -> str:
        return os.path.join(self.cache_dir, self._key(text) + ".json")

    def get(self, text: str) -> Optional[torch.Tensor]:
        path = self._path(text)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            vec = json.load(f)
        return torch.tensor(vec, dtype=torch.float32)

    def put(self, text: str, emb: torch.Tensor):
        path = self._path(text)
        if os.path.exists(path):
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(emb.detach().cpu().tolist(), f)


class OllamaEmbedder:
    def __init__(
        self, model: str, host: str = "http://localhost:11434", timeout: int = 120
    ):
        self.model = model
        self.url = host.rstrip("/") + "/api/embeddings"
        self.timeout = timeout
        self._dim: Optional[int] = None

    def embed_one(self, text: str) -> torch.Tensor:
        r = requests.post(
            self.url, json={"model": self.model, "prompt": text}, timeout=self.timeout
        )
        r.raise_for_status()
        data = r.json()
        vec = data.get("embedding", None)
        if vec is None:
            raise RuntimeError(f"No 'embedding' in response: {data}")
        if self._dim is None:
            self._dim = len(vec)
        return torch.tensor(vec, dtype=torch.float32)

    @property
    def dim(self) -> Optional[int]:
        return self._dim


def embed_sentences_cached(
    sentences: Sequence[str],
    embedder: OllamaEmbedder,
    cache: DiskEmbeddingCache,
    device: torch.device,
    embed_batch_size: int = 64,
) -> torch.Tensor:
    """
    Returns X: (N, d) on device. Uses per-sentence cache.
    """
    out: List[Optional[torch.Tensor]] = [None] * len(sentences)
    missing: List[str] = []
    missing_idx: List[int] = []

    # Cache lookup
    for i, s in enumerate(sentences):
        v = cache.get(s)
        if v is None:
            missing.append(s)
            missing_idx.append(i)
        else:
            out[i] = v

    # Embed missing
    if missing:
        for j in range(0, len(missing), embed_batch_size):
            chunk = missing[j : j + embed_batch_size]
            chunk_vecs = [embedder.embed_one(s) for s in chunk]
            for s, v in zip(chunk, chunk_vecs):
                cache.put(s, v)

            for local_k, v in enumerate(chunk_vecs):
                out_idx = missing_idx[j + local_k]
                out[out_idx] = v

    X = torch.stack([v for v in out if v is not None], dim=0).to(device)
    return X


# ============================================================
# 3) Optional: Global ABTT + z-score normalization (like your original)
# ============================================================


def abtt_and_zscore(X: torch.Tensor, n_remove: int = 2) -> torch.Tensor:
    """
    X: (N, d) on device
    Steps:
      - global center
      - PCA lowrank
      - remove top n_remove PCs (ABTT)
      - global z-score
    """
    X_centered = X - X.mean(dim=0, keepdim=True)

    n_samples, d = X_centered.shape
    q = min(max(n_remove + 2, n_remove), min(n_samples, d))
    U, S, V = torch.pca_lowrank(X_centered, q=q)  # V: (d, q)

    comps = V[:, :n_remove].T  # (n_remove, d)
    proj = X_centered @ comps.T
    X_debiased = X_centered - proj @ comps

    mean2 = X_debiased.mean(dim=0, keepdim=True)
    std2 = X_debiased.std(dim=0, keepdim=True, unbiased=False) + 1e-8
    return (X_debiased - mean2) / std2


# ============================================================
# 4) Your geometric experiment (SVD scoring + merge)
# ============================================================


def make_dataset_from_pool(seed: int, n: int, X_pool: torch.Tensor) -> torch.Tensor:
    """
    Build dataset X by sampling n rows (without replacement if possible) from X_pool.
    Then center columns.
    Returns X: (n_eff, d) on device.
    """
    rng_local = np.random.default_rng(seed)

    N_pool = X_pool.shape[0]
    n_eff = max(2, int(n))
    n_eff = min(n_eff, N_pool)

    if n_eff == N_pool:
        idx = np.arange(N_pool)
        rng_local.shuffle(idx)
    else:
        idx = rng_local.choice(N_pool, size=n_eff, replace=False)

    idx_t = torch.as_tensor(idx, device=X_pool.device, dtype=torch.long)
    X = X_pool[idx_t]
    X = X - X.mean(dim=0, keepdim=True)
    return X


def fit_V_on_rows(X: torch.Tensor, rows, r: int) -> torch.Tensor:
    rows_t = torch.as_tensor(rows, device=X.device, dtype=torch.long)
    Xtr = X[rows_t]
    U, S, Vh = torch.linalg.svd(Xtr, full_matrices=False)
    r_eff = min(r, Vh.shape[0])
    return Vh[:r_eff, :].T  # (d, r_eff)


def score_on_rows(X: torch.Tensor, rows, V: torch.Tensor) -> float:
    rows_t = torch.as_tensor(rows, device=X.device, dtype=torch.long)
    Xte = X[rows_t]
    P = V @ V.T
    Xproj = Xte @ P
    num = torch.norm(Xproj, p="fro") ** 2
    den = torch.norm(Xte, p="fro") ** 2 + 1e-12
    return (num / den).item()


def mean_score_on_run(V: torch.Tensor, X_list, splits) -> float:
    scores = []
    for j, Xj in enumerate(X_list):
        _, te = splits[j]
        scores.append(score_on_rows(Xj, te, V))
    return float(np.mean(scores))


def merge_subspaces(V1: torch.Tensor, V2: torch.Tensor, r: int) -> torch.Tensor:
    B = torch.cat([V1, V2], dim=1)
    U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    r_eff = min(r, U.shape[1])
    return U[:, :r_eff]


def run_one_experiment(
    global_seed_offset: int,
    X_pool: torch.Tensor,
    rng_global: np.random.Generator,
    K: int,
    r: int,
    n_min: int,
    n_max: int,
    train_frac: float,
):
    # 1) build datasets
    X_list = [
        make_dataset_from_pool(
            global_seed_offset + i, int(rng_global.integers(n_min, n_max)), X_pool
        )
        for i in range(K)
    ]

    # 2) train/test splits
    splits = []
    for X in X_list:
        n = X.shape[0]
        idx = rng_global.permutation(n)
        ntr = max(2, int(train_frac * n))
        tr, te = idx[:ntr], idx[ntr:]
        if len(te) == 0:
            te = idx[-1:]
            tr = idx[:-1]
        splits.append((tr, te))

    # 3) fit per-dataset subspaces
    V_list = [fit_V_on_rows(X_list[j], splits[j][0], r) for j in range(K)]

    # 4) replacements
    replacements = np.zeros(K, dtype=int)
    for j, Xj in enumerate(X_list):
        _, te = splits[j]
        native = score_on_rows(Xj, te, V_list[j])
        scores = [score_on_rows(Xj, te, V_list[i]) for i in range(K)]
        best_i = int(np.argmax(scores))
        # print(
        #     f"[run seed {global_seed_offset}] j={j}, native={native:.4f}, "
        #     f"best_i={best_i}, best_score={scores[best_i]:.4f}"
        # )
        if best_i != j and scores[best_i] > native + 1e-6:
            replacements[best_i] += 1

    top = int(np.argmax(replacements))
    return X_list, splits, V_list, replacements, top


# ============================================================
# 5) The entry point you asked for: run_training_example(...)
# ============================================================


def run_geometric_training_example(
    tsv_path: str,
    *,
    seed: int = 42,  # SAME seed as PyTorch experiments
    pair_subset_size: int = 50_000,  # sample this many PAIRS from TSV
    sentence_subset_size: int = 100_000,  # after expanding to sentences and shuffling, keep this many
    dedup_sentences: bool = True,
    ollama_model: str = "granite-embedding:278m",
    cache_dir: str = "./emb_cache_granite278m",
    do_abtt: bool = True,
    abtt_remove: int = 2,
    # geometric experiment params
    K: int = 100,
    r: int = 256,
    n_min: int = 10,
    n_max: int = 91,
    train_frac: float = 0.7,
    NUM_RUNS: int = 10,
    GOOD_TOL: float = 5e-3,
) -> torch.Tensor:
    """
    Returns:
      V_acc: (d, r_eff) universal subspace candidate on device.
    """
    set_all_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[info] Using device:", device)

    # ---- 1) Deterministic pair subset ----
    cfg = SplitConfig(subset_size=pair_subset_size, seed=seed)
    if cfg.subset_size is None:
        pairs = load_all_tsv_pairs(tsv_path)
    else:
        pairs = reservoir_sample_tsv_pairs(tsv_path, k=cfg.subset_size, seed=cfg.seed)

    # ---- 2) Expand into sentences, deterministic shuffle, take sentence subset ----
    sentence_pool, langs_in_pool, lang_counts = pairs_to_sentence_pool_with_langs(
        pairs,
        sentence_subset_size=sentence_subset_size,
        seed=seed,
        dedup=dedup_sentences,
    )

    print(f"[info] Sentence pool size = {len(sentence_pool)} (dedup={dedup_sentences})")
    print(f"[info] Languages in pool = {len(langs_in_pool)}")
    print("[info] Included languages (sorted):", langs_in_pool)

    # Optional: show top/bottom counts
    top10 = sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    bot10 = sorted(lang_counts.items(), key=lambda x: x[1])[:10]
    print("[info] Top-10 languages by sentence count:", top10)
    print("[info] Bottom-10 languages by sentence count:", bot10)

    # ---- 3) Embed sentence pool with Ollama + cache ----
    embedder = OllamaEmbedder(model=ollama_model)
    cache = DiskEmbeddingCache(cache_dir)
    X_pool = embed_sentences_cached(
        sentence_pool,
        embedder=embedder,
        cache=cache,
        device=device,
        embed_batch_size=64,
    )  # (N_pool, d)
    print(
        f"[info] Embedded pool X_pool shape = {tuple(X_pool.shape)} using {ollama_model}"
    )

    # ---- 4) Optional ABTT + z-score on the pool ----
    if do_abtt:
        print("[info] Applying ABTT + z-score on sentence pool...")
        X_pool = abtt_and_zscore(X_pool, n_remove=abtt_remove)

    # ---- 5) Run geometric meta-experiment and return V_acc ----
    rng_global = np.random.default_rng(seed)  # deterministic meta randomness

    V_acc = None
    acc_initialized = False

    for t in range(NUM_RUNS):
        print("\n" + "=" * 60)
        print(f"META-RUN {t}")
        seed_offset = 10000 * t  # same idea you used

        X_list, splits, V_list, replacements, top_idx = run_one_experiment(
            seed_offset,
            X_pool,
            rng_global,
            K=K,
            r=r,
            n_min=n_min,
            n_max=n_max,
            train_frac=train_frac,
        )
        V_new = V_list[top_idx]
        mean_new = mean_score_on_run(V_new, X_list, splits)

        print(f"[meta-run {t}] mean score of new top V_new: {mean_new:.4f}")
        print(f"[meta-run {t}] top_idx={top_idx}, replaced={replacements[top_idx]}")

        if not acc_initialized:
            V_acc = V_new
            acc_initialized = True
            print(f"[meta-run {t}] Initialized V_acc with V_new.")
            continue

        mean_acc = mean_score_on_run(V_acc, X_list, splits)
        print(f"[meta-run {t}] mean score of existing V_acc: {mean_acc:.4f}")

        diff = mean_new - mean_acc
        print(f"[meta-run {t}] mean_new - mean_acc = {diff:.6f}")

        if mean_new >= mean_acc - GOOD_TOL:
            print(f"[meta-run {t}] V_new is good enough; merging into V_acc.")
            V_acc = merge_subspaces(V_acc, V_new, r)
        else:
            print(f"[meta-run {t}] V_new too weak; keeping current V_acc.")

    print("\n" + "=" * 60)
    print("Finished meta-runs.")
    print("Final accumulated V_acc is your universal subspace candidate.")

    return V_acc


if __name__ == "__main__":
    # Example usage:
    V_acc = run_geometric_training_example(
        tsv_path="E:/thesis_work/nllb_sampled/merged.tsv",
        seed=42,  # same seed as your PyTorch pipeline
        pair_subset_size=10,  # sample pairs first
        sentence_subset_size=20,  # then sample sentences (src+tgt)
        dedup_sentences=True,
        ollama_model="granite-embedding:278m",
        cache_dir="./emb_cache_granite278m",
        do_abtt=True,
        abtt_remove=2,
        K=50,
        r=128,
        NUM_RUNS=5,
    )
    print("V_acc shape:", tuple(V_acc.shape))
