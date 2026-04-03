import hashlib
import json
import os
from json import JSONDecodeError
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class HFEmbedder:
    """
    Simple Hugging Face embedder.

    - embed_one(text) -> Tensor[d]
    - embed(sentences) -> Tensor[N, d]
    - __call__(sentences) -> Tensor[N, d]

    Uses mean pooling over the last hidden states.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_length: int = 512,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self._dim: Optional[int] = getattr(self.model.config, "hidden_size", None)

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @torch.no_grad()
    def embed_one(self, text: str) -> torch.Tensor:
        if text is None:
            raise ValueError("embed_one received None text")
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            raise ValueError("embed_one received empty/whitespace string")

        batch = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        outputs = self.model(**batch)
        emb = self._mean_pool(outputs.last_hidden_state, batch["attention_mask"])[0]

        if self.normalize:
            emb = F.normalize(emb, p=2, dim=0)

        if self._dim is None:
            self._dim = emb.shape[0]

        return emb.detach().cpu()

    @torch.no_grad()
    def embed(self, sentences: List[str]) -> torch.Tensor:
        if sentences is None:
            raise ValueError("embed received None sentences list")

        cleaned = []
        for i, s in enumerate(sentences):
            if s is None:
                raise ValueError(f"embed received None at index {i}")
            if not isinstance(s, str):
                s = str(s)
            s = s.strip()
            if not s:
                raise ValueError(f"embed received empty/whitespace string at index {i}")
            cleaned.append(s)

        if not cleaned:
            d = self._dim or 0
            return torch.empty((0, d), dtype=torch.float32)

        batch = self.tokenizer(
            cleaned,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        outputs = self.model(**batch)
        embs = self._mean_pool(outputs.last_hidden_state, batch["attention_mask"])

        if self.normalize:
            embs = F.normalize(embs, p=2, dim=1)

        if self._dim is None and embs.numel() > 0:
            self._dim = embs.shape[1]

        return embs.detach().cpu()

    @property
    def dim(self) -> Optional[int]:
        return self._dim

    def __call__(self, sentences: List[str]) -> torch.Tensor:
        return self.embed(sentences)


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

        try:
            with open(path, "r", encoding="utf-8") as f:
                vec = json.load(f)
        except (JSONDecodeError, OSError):
            try:
                os.remove(path)
            except OSError:
                pass
            return None

        return torch.tensor(vec, dtype=torch.float32)

    def put(self, text: str, emb: torch.Tensor):
        path = self._path(text)
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


@torch.no_grad()
def embed_sentences_in_batches(
    sentences: List[str],
    embed_fn: Callable[[List[str]], torch.Tensor],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """
    embed_fn: takes List[str] -> FloatTensor (B, d) on CPU or GPU.
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
    embedder: HFEmbedder,
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

    for i, s in enumerate(sentences):
        v = cache.get(s)
        if v is None:
            missing.append(s)
            missing_idx.append(i)
        else:
            out[i] = v

    if missing:
        for j in range(0, len(missing), embed_batch_size):
            chunk = missing[j : j + embed_batch_size]
            chunk_vecs = embedder.embed(chunk)  # batched HF embedding
            for s, v in zip(chunk, chunk_vecs):
                cache.put(s, v)

            for local_k, v in enumerate(chunk_vecs):
                out_idx = missing_idx[j + local_k]
                out[out_idx] = v

    X = torch.stack([v for v in out if v is not None], dim=0).to(device)
    return X