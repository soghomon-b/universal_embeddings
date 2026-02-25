import requests
import torch
from typing import List
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


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

    def get(self, text: str) -> Optional[torch.Tensor]:
        path = os.path.join(self.cache_dir, self._key(text) + ".json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            vec = json.load(f)
        return torch.tensor(vec, dtype=torch.float32)

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
