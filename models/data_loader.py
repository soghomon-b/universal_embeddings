import os
import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
from json import JSONDecodeError
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 1) TSV sampling + splitting
# -----------------------------


@dataclass
class SplitConfig:
    subset_size: Optional[int] = None  # e.g., 50000; None => use all
    train_frac: float = 0.90
    val_frac: float = 0.05
    test_frac: float = 0.05
    seed: int = 42

    def __post_init__(self):
        s = self.train_frac + self.val_frac + self.test_frac
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Fractions must sum to 1.0, got {s}")


def parse_parallel_tsv_line(line: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Expected formats:
      4 cols: src_lang \t tgt_lang \t src_sent \t tgt_sent
    Returns tuple or None if malformed.
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


def reservoir_sample_tsv(
    path: str,
    k: int,
    seed: int = 42,
    max_lines: Optional[int] = None,
) -> List[Tuple[str, str, str, str]]:
    """
    Streaming reservoir sample of k examples from a TSV file.
    Uses O(k) memory, suitable for 400k+ lines.
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
                j = rng.randrange(seen)  # 0..seen-1
                if j < k:
                    sample[j] = item

    if not sample:
        raise ValueError("No valid examples were parsed from the TSV.")
    return sample


def load_all_tsv(
    path: str,
    max_lines: Optional[int] = None,
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
    n_test = n - n_train - n_val

    train = ex[:n_train]
    val = ex[n_train : n_train + n_val]
    test = ex[n_train + n_val :]
    assert len(test) == n_test
    return train, val, test


# -----------------------------
# 2) Dataset + collate
# -----------------------------


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
    # batch is list of tuples
    src_langs, tgt_langs, s1, s2 = zip(*batch)
    return list(src_langs), list(tgt_langs), list(s1), list(s2)


# -----------------------------
# 3) Embedding wrapper + batch generator for your train_infonce
# -----------------------------


@torch.no_grad()
def embed_sentences_in_batches(
    sentences: List[str],
    embed_fn: Callable[[List[str]], torch.Tensor],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """
    embed_fn: takes List[str] -> FloatTensor (B, d) on CPU or GPU (your choice).
    We move to device after embedding to keep it flexible.
    """
    embs = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i : i + batch_size]
        e = embed_fn(chunk)
        if not torch.is_tensor(e):
            e = torch.tensor(e, dtype=torch.float32)
        e = e.float().to(device)
        embs.append(e)
    return torch.cat(embs, dim=0)


def make_infonce_batches_from_loader(
    loader: DataLoader,
    embed_fn_src: Callable[[List[str]], torch.Tensor],
    embed_fn_tgt: Callable[[List[str]], torch.Tensor],
    device: str = "cpu",
    embed_batch_size: int = 64,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Yields (E1, E2) tensors where row i matches row i.
    Negatives are all other rows in the batch (in-batch negatives).
    """
    for src_langs, tgt_langs, s1, s2 in loader:
        # If you want language-aware embedding, you can incorporate langs inside embed_fn.
        E1 = embed_sentences_in_batches(s1, embed_fn_src, embed_batch_size, device)
        E2 = embed_sentences_in_batches(s2, embed_fn_tgt, embed_batch_size, device)
        yield E1, E2


# -----------------------------
# 4) Example embed_fn options
# -----------------------------


def make_dummy_embedder(
    d: int = 768, seed: int = 0
) -> Callable[[List[str]], torch.Tensor]:
    """
    Placeholder embedder to test the pipeline without transformers.
    Replace with a real encoder (e.g., LaBSE, LASER, mpnet, etc.)
    """
    rng = torch.Generator().manual_seed(seed)

    def _embed(sents: List[str]) -> torch.Tensor:
        # deterministic-ish by length to make debugging easier
        lens = torch.tensor([len(s) for s in sents], dtype=torch.float32).unsqueeze(1)
        noise = torch.randn(len(sents), d, generator=rng) * 0.01
        base = lens.repeat(1, d) / 100.0
        return (base + noise).float()

    return _embed


# -----------------------------
# 5) End-to-end "prepare data" utility
# -----------------------------


def prepare_parallel_data(
    tsv_path: str,
    cfg: SplitConfig,
    batch_size_pairs: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
    max_lines: Optional[int] = None,  # for quick dev
):
    """
    Returns train_loader, val_loader, test_loader
    Each yields (src_langs, tgt_langs, src_sents, tgt_sents)
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(tsv_path)

    if cfg.subset_size is None:
        examples = load_all_tsv(tsv_path, max_lines=max_lines)
    else:
        examples = reservoir_sample_tsv(
            tsv_path, k=cfg.subset_size, seed=cfg.seed, max_lines=max_lines
        )

    train_ex, val_ex, test_ex = make_splits(examples, cfg)

    train_ds = ParallelTextDataset(train_ex)
    val_ds = ParallelTextDataset(val_ex)
    test_ds = ParallelTextDataset(test_ex)

    # Note: We shuffle at the DataLoader too (helps training)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_pairs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_parallel,
        drop_last=True,  # important for stable in-batch negatives
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_pairs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_parallel,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size_pairs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_parallel,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


import requests
import torch
from typing import List


class OllamaEmbedder:
    """
    Calls Ollama local embeddings endpoint:
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
        self._dim = None  # discovered on first call

    def __call__(self, sents: List[str]) -> torch.Tensor:
        embs = []
        for s in sents:
            r = requests.post(
                self.url,
                json={"model": self.model, "prompt": s},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            vec = data.get("embedding", None)
            if vec is None:
                raise RuntimeError(f"No 'embedding' in response: {data}")
            if self._dim is None:
                self._dim = len(vec)
            embs.append(torch.tensor(vec, dtype=torch.float32))
        return torch.stack(embs, dim=0)

    @property
    def dim(self):
        return self._dim


import hashlib
import os
import json
from typing import Dict, Optional


class DiskEmbeddingCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, key: str):
        path = self._key_to_path(key)  # whatever you already do
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
        path = os.path.join(self.cache_dir, self._key(text) + ".json")
        if os.path.exists(path):
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(emb.detach().cpu().tolist(), f)


class CachedEmbedder:
    def __init__(self, base_embedder, cache: DiskEmbeddingCache):
        self.base = base_embedder
        self.cache = cache

    def __call__(self, sents):
        out = []
        missing = []
        missing_idx = []

        for i, s in enumerate(sents):
            v = self.cache.get(s)
            if v is None:
                missing.append(s)
                missing_idx.append(i)
                out.append(None)
            else:
                out.append(v)

        if missing:
            new_vecs = self.base(missing)  # (M, d)
            for j, s in enumerate(missing):
                self.cache.put(s, new_vecs[j])
            for pos, vec in zip(missing_idx, new_vecs):
                out[pos] = vec

        return torch.stack(out, dim=0)


def make_pairwise_batches_from_loader(
    loader: DataLoader,
    embed_fn_src: Callable[[List[str]], torch.Tensor],
    embed_fn_tgt: Callable[[List[str]], torch.Tensor],
    device: str = "cpu",
    embed_batch_size: int = 64,
    neg_ratio: float = 1.0,  # 1.0 => same number of negs as pos
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Yields (y, E1, E2).
    Positives: aligned parallel pairs => y=+1
    Negatives: mismatched pairs via shuffling => y=-1
    """
    for src_langs, tgt_langs, s1, s2 in loader:
        E1 = embed_sentences_in_batches(s1, embed_fn_src, embed_batch_size, device)
        E2 = embed_sentences_in_batches(s2, embed_fn_tgt, embed_batch_size, device)

        B = E1.size(0)

        # positives
        y_pos = torch.ones(B, device=E1.device, dtype=torch.float32)

        # negatives: shuffle targets
        perm = torch.randperm(B, device=E1.device)
        E2_neg = E2[perm]
        y_neg = -torch.ones(B, device=E1.device, dtype=torch.float32)

        # optionally subsample negatives
        if neg_ratio < 1.0:
            m = max(1, int(B * neg_ratio))
            idx = torch.randperm(B, device=E1.device)[:m]
            E1_neg = E1[idx]
            E2_neg = E2_neg[idx]
            y_neg = y_neg[idx]
        else:
            E1_neg = E1

        # concatenate
        y = torch.cat([y_pos, y_neg], dim=0)          # (B + M,)
        E1_all = torch.cat([E1, E1_neg], dim=0)       # (B + M, d)
        E2_all = torch.cat([E2, E2_neg], dim=0)       # (B + M, d)

        yield y, E1_all, E2_all

