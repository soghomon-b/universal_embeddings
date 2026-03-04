"""
supcon_ollama_train.py

End-to-end script that:
1) Loads a parallel TSV (src_lang \t tgt_lang \t src_sent \t tgt_sent)
2) Samples a subset (reservoir sampling) + shuffles + splits train/val/test
3) Uses Ollama embeddings (granite-embedding:278m) with a disk cache
4) Builds *SupCon* batches from parallel pairs (2 views per pair in each batch)
5) Trains a LinearProjector with Supervised Contrastive Loss (SupCon)

Notes:
- This does NOT use your old pairwise CosineEmbeddingLoss "train" function.
  SupCon needs (E, labels) batches, not (E1, E2).
- Negatives are in-batch: any different label is treated as a negative.
- Requires: pip install requests torch
- Requires Ollama running locally (default http://localhost:11434) and model pulled:
    ollama pull granite-embedding:278m
"""

import os
import json
from json import JSONDecodeError
import random
import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0) Projector (replace with your own import if you prefer)
# ============================================================


class LinearProjector(nn.Module):
    def __init__(self, d: int, k: int):
        super().__init__()
        self.proj = nn.Linear(d, k, bias=False)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        z = self.proj(e)
        return F.normalize(z, dim=-1)


# ============================================================
# 1) TSV parsing / subset sampling / splitting
# ============================================================


@dataclass
class SplitConfig:
    subset_size: Optional[int] = None  # None => use all
    train_frac: float = 1
    val_frac: float = 0.00
    test_frac: float = 0.00
    seed: int = 42

    def __post_init__(self):
        s = self.train_frac + self.val_frac + self.test_frac
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Fractions must sum to 1.0, got {s}")


def parse_parallel_tsv_line(line: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Expected: src_lang \t tgt_lang \t src_sent \t tgt_sent
    """
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


def extract_languages(examples: Sequence[Tuple[str, str, str, str]]) -> List[str]:
    langs = set()
    for src_lang, tgt_lang, _, _ in examples:
        langs.add(src_lang)
        langs.add(tgt_lang)
    return sorted(langs)


def reservoir_sample_tsv(
    path: str,
    k: int,
    seed: int = 42,
    max_lines: Optional[int] = None,
) -> List[Tuple[str, str, str, str]]:
    """
    Streaming reservoir sampling (uniform) in O(k) memory.
    """
    rng = random.Random(seed)
    sample: List[Tuple[str, str, str, str]] = []
    seen = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if max_lines is not None and line_idx >= max_lines:
                break
            item = parse_parallel_tsv_line(line)
            if item is None:
                continue

            seen += 1
            if len(sample) < k:
                sample.append(item)
            else:
                j = rng.randrange(seen)
                if j < k:
                    sample[j] = item

    if not sample:
        raise ValueError("No valid examples were parsed from the TSV.")
    return sample


def load_all_tsv(
    path: str, max_lines: Optional[int] = None
) -> List[Tuple[str, str, str, str]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if max_lines is not None and line_idx >= max_lines:
                break
            item = parse_parallel_tsv_line(line)
            if item is not None:
                out.append(item)
    if not out:
        raise ValueError("No valid examples were parsed from the TSV.")
    return out


def make_splits(
    examples: Sequence[Tuple[str, str, str, str]],
    cfg: SplitConfig,
) -> Tuple[
    List[Tuple[str, str, str, str]],
    List[Tuple[str, str, str, str]],
    List[Tuple[str, str, str, str]],
]:
    rng = random.Random(cfg.seed)
    ex = list(examples)
    rng.shuffle(ex)

    n = len(ex)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    # remainder to test
    train = ex[:n_train]
    val = ex[n_train : n_train + n_val]
    test = ex[n_train + n_val :]
    return train, val, test


class ParallelTextDataset(Dataset):
    """
    Stores (src_lang, tgt_lang, src_sent, tgt_sent)
    """

    def __init__(self, examples: Sequence[Tuple[str, str, str, str]]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def collate_parallel(batch):
    src_langs, tgt_langs, s1, s2 = zip(*batch)
    return list(src_langs), list(tgt_langs), list(s1), list(s2)


def prepare_parallel_data(
    tsv_path: str,
    cfg: SplitConfig,
    batch_size_pairs: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
    max_lines: Optional[int] = None,
):
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(tsv_path)

    if cfg.subset_size is None:
        examples = load_all_tsv(tsv_path, max_lines=max_lines)
    else:
        examples = reservoir_sample_tsv(
            tsv_path, k=cfg.subset_size, seed=cfg.seed, max_lines=max_lines
        )

    languages = extract_languages(examples)

    train_ex, val_ex, test_ex = make_splits(examples, cfg)

    train_loader = DataLoader(
        ParallelTextDataset(train_ex),
        batch_size=batch_size_pairs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_parallel,
        drop_last=True,
    )
    val_loader = DataLoader(
        ParallelTextDataset(val_ex),
        batch_size=batch_size_pairs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_parallel,
        drop_last=False,
    )
    test_loader = DataLoader(
        ParallelTextDataset(test_ex),
        batch_size=batch_size_pairs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_parallel,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader, languages


# ============================================================
# 2) Ollama embeddings + caching
# ============================================================


class OllamaEmbedder:
    """
    Calls Ollama embeddings endpoint:
      POST http://localhost:11434/api/embeddings
      JSON: { "model": "...", "prompt": "..." }
    Returns FloatTensor (B, d)
    """

    def __init__(
        self,
        model: str = "granite-embedding:278m",
        host: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model = model
        self.url = host.rstrip("/") + "/api/embeddings"
        self.timeout = timeout
        self._dim: Optional[int] = None

    def __call__(self, sents: List[str]) -> torch.Tensor:
        embs = []
        for s in sents:
            r = requests.post(
                self.url, json={"model": self.model, "prompt": s}, timeout=self.timeout
            )
            r.raise_for_status()
            data = r.json()
            vec = data.get("embedding")
            if vec is None:
                raise RuntimeError(f"No 'embedding' in response: {data}")
            if self._dim is None:
                self._dim = len(vec)
            embs.append(torch.tensor(vec, dtype=torch.float32))
        return torch.stack(embs, dim=0)

    @property
    def dim(self) -> Optional[int]:
        return self._dim


class DiskEmbeddingCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _path(self, text: str) -> str:
        return os.path.join(self.cache_dir, self._key(text) + ".json")

    def get(self, text: str) -> Optional[torch.Tensor]:
        path = self._path(text)
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (JSONDecodeError, OSError) as e:
            # corrupted cache entry -> delete and treat as miss
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def put(self, text: str, emb: torch.Tensor):
        path = self._path(text)
        if os.path.exists(path):
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(emb.detach().cpu().tolist(), f)


class CachedEmbedder:
    def __init__(
        self,
        base_embedder: Callable[[List[str]], torch.Tensor],
        cache: DiskEmbeddingCache,
    ):
        self.base = base_embedder
        self.cache = cache

    def __call__(self, sents: List[str]) -> torch.Tensor:
        out: List[Optional[torch.Tensor]] = [None] * len(sents)
        missing: List[str] = []
        missing_idx: List[int] = []

        for i, s in enumerate(sents):
            v = self.cache.get(s)
            if v is None:
                missing.append(s)
                missing_idx.append(i)
            else:
                out[i] = v

        if missing:
            new_vecs = self.base(missing)  # (M, d)
            for j, s in enumerate(missing):
                self.cache.put(s, new_vecs[j])
            for idx, vec in zip(missing_idx, new_vecs):
                out[idx] = vec

        return torch.stack([v for v in out if v is not None], dim=0)


@torch.no_grad()
def embed_sentences_in_batches(
    sentences: List[str],
    embed_fn: Callable[[List[str]], torch.Tensor],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    embs = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i : i + batch_size]
        e = embed_fn(chunk)
        if not torch.is_tensor(e):
            e = torch.tensor(e, dtype=torch.float32)
        embs.append(e.float().to(device))
    return torch.cat(embs, dim=0)


# ============================================================
# 3) SupCon batches from parallel data
# ============================================================


def make_supcon_batches_from_parallel_loader(
    loader: DataLoader,
    embed_src: Callable[[List[str]], torch.Tensor],
    embed_tgt: Callable[[List[str]], torch.Tensor],
    device: str = "cpu",
    embed_batch_size: int = 64,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    """
    For each batch of B parallel pairs:
      E1: (B,d) embeddings for src sentences
      E2: (B,d) embeddings for tgt sentences

    We create 2B "views":
      E = [E1; E2]        shape (2B,d)
      labels = [0..B-1, 0..B-1]  shape (2B,)
    so each pair-id appears twice => at least one positive per anchor.
    """
    for _, _, s1, s2 in loader:
        E1 = embed_sentences_in_batches(
            s1, embed_src, embed_batch_size, device
        )  # (B,d)
        E2 = embed_sentences_in_batches(
            s2, embed_tgt, embed_batch_size, device
        )  # (B,d)

        if E1.size(0) != E2.size(0):
            raise RuntimeError(f"Batch size mismatch: {E1.size(0)} vs {E2.size(0)}")

        B = E1.size(0)
        E = torch.cat([E1, E2], dim=0)  # (2B,d)
        labels = torch.arange(B, device=device, dtype=torch.long).repeat(2)  # (2B,)

        yield E, labels


# ============================================================
# 4) SupCon loss + training (EDITED)
# ============================================================


def supcon_loss(
    z: torch.Tensor, labels: torch.Tensor, tau: float = 0.07
) -> Optional[torch.Tensor]:
    """
    z: (N, k) normalized embeddings
    labels: (N,) int labels; same label => positives

    Returns:
      loss tensor, or None if batch contains no positives (shouldn't happen with our batching).
    """
    device = z.device
    N = z.size(0)
    if N < 2:
        return None

    labels = labels.view(-1, 1)  # (N,1)
    mask = labels == labels.T  # (N,N) bool

    logits = (z @ z.T) / tau

    # remove self-contrast
    self_mask = torch.eye(N, device=device, dtype=torch.bool)
    mask = mask.masked_fill(self_mask, False)  # positives exclude self
    logits = logits.masked_fill(self_mask, -1e9)  # exclude self from denominator

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)  # (N,N)

    pos_counts = mask.sum(dim=1)  # (N,)
    valid = pos_counts > 0
    if valid.sum() == 0:
        return None

    mean_log_prob_pos = (mask[valid] * log_prob[valid]).sum(dim=1) / pos_counts[valid]
    return -mean_log_prob_pos.mean()


def train_supcon(
    batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    d: int,
    k: int,
    lr: float = 1e-3,
    epochs: int = 5,
    tau: float = 0.07,
    device: str = "cpu",
) -> LinearProjector:
    """
    batches yields (E, labels):
      E: (N,d) embeddings (here N=2B for B pairs)
      labels: (N,) int group ids (pair ids inside the batch)
    """
    model = LinearProjector(d, k).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"running supcon training with {epochs}")
    for _ in range(epochs):
        for E, labels in batches:
            if not torch.is_tensor(E):
                E = torch.tensor(E, dtype=torch.float32)
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels, dtype=torch.long)

            E = E.to(device).float()
            labels = labels.to(device).long()

            optimizer.zero_grad()
            z = model(E)  # (N,k)
            loss = supcon_loss(z, labels, tau=tau)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()

    return model


# ============================================================
# 5) Run training example (EDITED to SupCon)
# ============================================================


def run_supcon_training_example(
    tsv_path: str,
    seed: int,
    subset_size: int = 50000,
    d: Optional[int] = None,  # if None, auto-detect from embedder first call
    k: int = 256,
    batch_size_pairs: int = 256,
    epochs: int = 5,
    lr: float = 1e-3,
    tau: float = 0.07,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    cfg = SplitConfig(
        subset_size=subset_size,
        train_frac=0.90,
        val_frac=0.05,
        test_frac=0.05,
        seed=seed,
    )
    train_loader, val_loader, test_loader, languages = prepare_parallel_data(
        tsv_path,
        cfg,
        batch_size_pairs=batch_size_pairs,
        num_workers=0,
        pin_memory=(device.startswith("cuda")),
    )

    # Ollama embedder + cache (same model for src & tgt)
    embed_base = OllamaEmbedder(model="granite-embedding:278m")
    cache = DiskEmbeddingCache("./emb_cache_granite278m")
    embed_src = CachedEmbedder(embed_base, cache)
    embed_tgt = embed_src

    # Auto-detect d (embedding dimension) if not provided:
    if d is None:
        probe = embed_src(["dimension probe sentence"])
        d = probe.shape[1]
        print(f"[info] Detected embedding dimension d={d}")

    # SupCon batches from parallel pairs
    train_batches = make_supcon_batches_from_parallel_loader(
        train_loader, embed_src, embed_tgt, device=device, embed_batch_size=64
    )

    model = train_supcon(
        train_batches, d=d, k=k, lr=lr, epochs=epochs, tau=tau, device=device
    )

    return model, languages


if __name__ == "__main__":
    # Example usage:
    model, languages = run_supcon_training_example(
        tsv_path="",
        subset_size=10,  # small for smoke test
        d=None,  # auto-detect from Ollama embeddings
        k=11,
        batch_size_pairs=16,
        epochs=2,
        lr=1e-3,
        tau=0.07,
    )
    print(model)
