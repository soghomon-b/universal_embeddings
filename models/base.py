"""
base_retrieval_from_tsv.py

Goal
----
Compute *base* cross-lingual retrieval performance from a parallel TSV
(src_lang \t tgt_lang \t src_sent \t tgt_sent), using raw encoder embeddings.

This is the "do nothing" baseline:
  - embed src and tgt sentences
  - retrieve tgt for each src by cosine similarity (and optionally reverse)
  - report Accuracy@1, MRR, Recall@k

Determinism:
  - subset of pairs is deterministic via reservoir sampling with seed
  - optional dedup is deterministic
"""

from __future__ import annotations

import os
import json
from json import JSONDecodeError
import random
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Set

import numpy as np
import requests
import torch

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# ============================================================
# 0) Seed helpers
# ============================================================

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1) TSV loading + deterministic reservoir sampling
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


def reservoir_sample_tsv_pairs(path: str, k: int, seed: int) -> List[Tuple[str, str, str, str]]:
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
    out: List[Tuple[str, str, str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = parse_parallel_tsv_line(line)
            if item is not None:
                out.append(item)
    if not out:
        raise ValueError("No valid pairs parsed from TSV.")
    return out


def deterministic_dedup_pairs(
    pairs: Sequence[Tuple[str, str, str, str]],
    dedup_on: str = "both",  # "src" | "tgt" | "both"
) -> List[Tuple[str, str, str, str]]:
    """
    Deterministic dedup preserving first occurrence.
    Useful if your TSV has repeated sentences.
    """
    seen_src: Set[str] = set()
    seen_tgt: Set[str] = set()
    out: List[Tuple[str, str, str, str]] = []

    for sl, tl, s, t in pairs:
        ok = True
        if dedup_on in ("src", "both"):
            if s in seen_src:
                ok = False
            else:
                seen_src.add(s)
        if dedup_on in ("tgt", "both"):
            if t in seen_tgt:
                ok = False
            else:
                seen_tgt.add(t)

        if ok:
            out.append((sl, tl, s, t))

    if not out:
        raise ValueError("All pairs removed by dedup; try dedup_on=None or bigger subset.")
    return out


# ============================================================
# 2) HuggingFace embeddings with disk cache
# ============================================================

class DiskEmbeddingCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text)}")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _path(self, text: str) -> str:
        return os.path.join(self.cache_dir, self._key(text) + ".json")

    def get(self, text: str):
        path = self._path(text)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return np.array(json.load(f), dtype=np.float32)

    def set(self, text: str, vec):
        path = self._path(text)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(np.asarray(vec).tolist(), f)

    def put(self, text: str, vec):
        self.set(text, vec)

class CachedEmbedder:
    def __init__(self, base, cache):
        self.base = base
        self.cache = cache

    def __call__(self, texts):
        single_input = False
        if isinstance(texts, str):
            texts = [texts]
            single_input = True

        results = [None] * len(texts)
        missing_idx = []
        missing_texts = []

        for i, text in enumerate(texts):
            v = self.cache.get(text)
            if v is not None:
                results[i] = v
            else:
                missing_idx.append(i)
                missing_texts.append(text)

        if missing_texts:
            new_vecs = self.base(missing_texts)

            for i, text, vec in zip(missing_idx, missing_texts, new_vecs):
                self.cache.set(text, vec)
                results[i] = vec

        if single_input:
            return results[0]

        return np.stack(results, axis=0)


# ============================================================
# 2) Local HF embedder (replaces OllamaEmbedder)
# ============================================================
from transformers import AutoTokenizer, AutoModel
class HFEmbedder:
    """
    Simple Hugging Face embedder using mean pooling over the last hidden state.
    Works on Narval without Ollama.
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        max_length: int = 512,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        self._dim = self.model.config.hidden_size

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @torch.no_grad()
    def embed_batch(self, texts: Sequence[str]) -> torch.Tensor:
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)

        emb = self._mean_pool(outputs.last_hidden_state, batch["attention_mask"])

        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)

        return emb

    @torch.no_grad()
    def embed_one(self, text: str) -> torch.Tensor:
        return self.embed_batch([text])[0].detach().cpu()

    @property
    def dim(self) -> int:
        return self._dim



def embed_texts_cached(
    texts: Sequence[str],
    embedder: HFEmbedder,
    cache: DiskEmbeddingCache,
    device: torch.device,
    embed_batch_size: int = 64,
) -> torch.Tensor:
    """
    Returns E: (N, d) on device. Uses per-text cache.
    """
    out: List[Optional[torch.Tensor]] = [None] * len(texts)
    missing: List[str] = []
    missing_idx: List[int] = []

    for i, s in enumerate(texts):
        v = cache.get(s)
        if v is None:
            missing.append(s)
            missing_idx.append(i)
        else:
            out[i] = v

    if missing:
        for j in range(0, len(missing), embed_batch_size):
            chunk = missing[j : j + embed_batch_size]
            chunk_vecs = [embedder.embed_one(s) for s in chunk]
            for s, v in zip(chunk, chunk_vecs):
                cache.put(s, v)
            for local_k, v in enumerate(chunk_vecs):
                out_idx = missing_idx[j + local_k]
                out[out_idx] = v

    E = torch.stack([v for v in out if v is not None], dim=0).to(device)
    return E


# ============================================================
# 3) Optional ABTT + z-score normalization (same as your script)
# ============================================================

def abtt_and_zscore(X: torch.Tensor, n_remove: int = 2) -> torch.Tensor:
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
# 4) Retrieval metrics
# ============================================================

def l2_normalize(X: torch.Tensor) -> torch.Tensor:
    return X / (X.norm(dim=1, keepdim=True) + 1e-12)


@torch.no_grad()
def retrieval_metrics(
    Q: torch.Tensor,   # (N, d) queries
    D: torch.Tensor,   # (N, d) docs (aligned index-wise: i matches i)
    ks: Sequence[int] = (1, 3, 5),
    batch_size: int = 512,
) -> Dict[str, object]:
    """
    Brute-force cosine retrieval in batches.
    Assumes true match for query i is doc i.
    """
    Qn = l2_normalize(Q)
    Dn = l2_normalize(D)

    N = Qn.shape[0]
    ks = sorted(set(int(k) for k in ks))
    maxk = max(ks)

    correct_at_k = {k: 0 for k in ks}
    reciprocal_ranks = []

    # pretranspose for faster matmul
    Dt = Dn.T  # (d, N)

    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        qb = Qn[start:end]  # (b, d)
        sims = qb @ Dt      # (b, N)

        # ranks: find position of the true index within sorted sims desc
        # We can compute rank by counting how many scores are > true_score
        true_idx = torch.arange(start, end, device=Q.device)
        true_scores = sims[torch.arange(end - start, device=Q.device), true_idx]

        # rank = 1 + number of docs with score strictly greater than true_score
        # (ties: this makes rank optimistic; if you care, handle ties explicitly)
        greater = (sims > true_scores.unsqueeze(1)).sum(dim=1)
        rank = greater + 1  # (b,)
        reciprocal_ranks.append((1.0 / rank.float()).cpu())

        for k in ks:
            correct_at_k[k] += (rank <= k).sum().item()

    mrr = torch.cat(reciprocal_ranks).mean().item()
    recall = {k: correct_at_k[k] / N for k in ks}

    return {
        "N": N,
        "Accuracy@1": recall.get(1, correct_at_k.get(1, 0) / N),
        "MRR": mrr,
        "Recall@k": recall,
    }


# ============================================================
# 5) Entry point: run_base_retrieval_example(...)
# ============================================================

def run_base_retrieval_example(
    tsv_path: str,
    *,
    seed: int = 42,
    pair_subset_size: Optional[int] = 50_000,   # None => use all
    dedup_pairs: bool = True,
    dedup_on: str = "both",                     # "src" | "tgt" | "both"
    ollama_model: str = "llama3.1:8b",
    cache_dir: str = "./emb_cache_llama8b",
    do_abtt: bool = False,                      # True => ABTT+zscore baseline
    abtt_remove: int = 2,
    eval_both_directions: bool = True,          # src->tgt and tgt->src
    ks: Sequence[int] = (1, 3, 5),
) -> Dict[str, object]:
    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[info] Using device:", device)

    # ---- load / sample pairs deterministically ----
    cfg = SplitConfig(subset_size=pair_subset_size, seed=seed)
    if cfg.subset_size is None:
        pairs = load_all_tsv_pairs(tsv_path)
    else:
        pairs = reservoir_sample_tsv_pairs(tsv_path, k=cfg.subset_size, seed=cfg.seed)

    if dedup_pairs:
        pairs = deterministic_dedup_pairs(pairs, dedup_on=dedup_on)

    print(f"[info] Using {len(pairs)} pairs (dedup={dedup_pairs}, dedup_on={dedup_on})")

    src_sents = [p[2] for p in pairs]
    tgt_sents = [p[3] for p in pairs]

    # ---- embed with cache ----
    embedder = HFEmbedder(model_name=ollama_model, device=device)
    cache = DiskEmbeddingCache(cache_dir)

    print("[info] Embedding source sentences...")
    E_src = embed_texts_cached(src_sents, embedder, cache, device=device, embed_batch_size=64)

    print("[info] Embedding target sentences...")
    E_tgt = embed_texts_cached(tgt_sents, embedder, cache, device=device, embed_batch_size=64)
    X = torch.cat([E_src, E_tgt], dim=0)
    # ---- optional ABTT+zscore (applied jointly is usually better) ----
    if do_abtt:
        print("[info] Applying ABTT + z-score (jointly over src+tgt)...")
        X = abtt_and_zscore(X, n_remove=abtt_remove)
    
    return X


if __name__ == "__main__":
    res = run_base_retrieval_example(
        tsv_path="YOUR_PARALLEL.tsv",
        seed=42,
        pair_subset_size=50_000,
        dedup_pairs=True,
        dedup_on="both",
        ollama_model="llama3.1:8b",
        cache_dir="./emb_cache_llama8b",
        do_abtt=False,           # True gives "base+ABTT" baseline
        abtt_remove=2,
        eval_both_directions=True,
        ks=(1, 3, 5),
    )