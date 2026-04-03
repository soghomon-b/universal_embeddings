import os
import random
import hashlib
import json
from json import JSONDecodeError
from dataclasses import dataclass
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1) TSV parsing + splitting
# ============================================================

@dataclass
class SplitConfig:
    subset_size: Optional[int] = None
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
    Expected format:
        src_lang \t tgt_lang \t src_sent \t tgt_sent
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


def extract_languages(
    examples: Sequence[Tuple[str, str, str, str]]
) -> List[str]:
    langs = set()
    for src_lang, tgt_lang, _, _ in examples:
        langs.add(src_lang)
        langs.add(tgt_lang)
    return sorted(langs)


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
    val = ex[n_train:n_train + n_val]
    test = ex[n_train + n_val:]
    assert len(test) == n_test
    return train, val, test


# ============================================================
# 2) Dataset + collate
# ============================================================

class ParallelTextDataset(Dataset):
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
    """
    Returns:
        train_loader, val_loader, test_loader, languages
    """
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
# 3) Embedding helpers
# ============================================================

@torch.no_grad()
def embed_sentences_in_batches(
    sentences: List[str],
    embed_fn: Callable[[List[str]], torch.Tensor],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    embs = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i:i + batch_size]
        e = embed_fn(chunk)
        if not torch.is_tensor(e):
            e = torch.tensor(e, dtype=torch.float32)
        e = e.float().to(device)
        embs.append(e)
    return torch.cat(embs, dim=0)


import os
import json
import hashlib
from json import JSONDecodeError
from typing import Optional, Sequence, List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ============================================================
# 1) Disk cache (mostly unchanged)
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


# ============================================================
# 2) Local HF embedder (replaces OllamaEmbedder)
# ============================================================

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


# ============================================================
# 3) Cached embedding function
# ============================================================

def embed_sentences_cached(
    sentences: Sequence[str],
    embedder: HFEmbedder,
    cache: DiskEmbeddingCache,
    device: torch.device,
    embed_batch_size: int = 64,
) -> torch.Tensor:
    """
    Returns X: (N, d) on device. Uses per-sentence cache.
    Missing sentences are embedded in GPU batches.
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

    # Embed missing in batches
    if missing:
        for j in range(0, len(missing), embed_batch_size):
            chunk = missing[j : j + embed_batch_size]

            chunk_vecs = embedder.embed_batch(chunk).detach().cpu()

            for s, v in zip(chunk, chunk_vecs):
                cache.put(s, v)

            for local_k, v in enumerate(chunk_vecs):
                out_idx = missing_idx[j + local_k]
                out[out_idx] = v

    X = torch.stack([v for v in out if v is not None], dim=0).to(device)
    return X

class CachedEmbedder:
    def __init__(self, base, cache: DiskEmbeddingCache):
        self.base = base
        self.cache = cache

    def __call__(self, text: str) -> torch.Tensor:
        v = self.cache.get(text)
        if v is not None:
            return v

        v = self.base.embed_one(text)
        self.cache.put(text, v)
        return v
# ============================================================
# 4) Multilingual masked GCCA
# ============================================================

class MaskedGCCAProjector(nn.Module):
    """
    One linear projector per language into a shared k-dim space.
    Supports missing views by using row indices.
    """

    def __init__(self, dims_by_lang: Dict[str, int], k: int):
        super().__init__()
        self.k = k
        self.langs = sorted(dims_by_lang.keys())

        self.projs = nn.ModuleDict({
            lang: nn.Linear(dims_by_lang[lang], k, bias=False)
            for lang in self.langs
        })

        # learned during fitting for centering
        self.register_buffer("_dummy", torch.tensor(0.0), persistent=False)
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
    def fit_gcca(
        self,
        views: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        n_total: int,
        epochs: int = 10,
        reg: float = 1e-4,
        verbose: bool = True,
    ):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        if not views:
            raise ValueError("No views provided to GCCA.")

        G = torch.randn(n_total, self.k, device=device, dtype=dtype)
        G = F.normalize(G, dim=-1)

        for epoch in range(epochs):
            if verbose:
                print(f"[GCCA] epoch {epoch + 1}/{epochs}")

            for lang in self.langs:
                if lang not in views:
                    continue

                X, idx = views[lang]
                X = X.to(device=device, dtype=dtype)
                idx = idx.to(device=device)

                mean = X.mean(dim=0, keepdim=True)
                Xc = X - mean
                self.means_by_lang[lang] = mean.detach().clone()

                G_sub = G[idx]

                d = Xc.shape[1]
                XtX = Xc.T @ Xc + reg * torch.eye(d, device=device, dtype=dtype)
                XtG = Xc.T @ G_sub
                W = torch.linalg.solve(XtX, XtG)

                self.projs[lang].weight.copy_(W.T)

            G_new = torch.zeros_like(G)
            counts = torch.zeros(n_total, 1, device=device, dtype=dtype)

            for lang in self.langs:
                if lang not in views:
                    continue

                X, idx = views[lang]
                X = X.to(device=device, dtype=dtype)
                idx = idx.to(device=device)

                mean = self.means_by_lang[lang].to(device=device, dtype=dtype)
                Xc = X - mean

                Z = self.projs[lang](Xc)
                G_new.index_add_(0, idx, Z)
                counts.index_add_(0, idx, torch.ones(idx.shape[0], 1, device=device, dtype=dtype))

            mask = counts.squeeze(-1) > 0
            G_new[mask] = G_new[mask] / counts[mask]
            G_new[mask] = F.normalize(G_new[mask], dim=-1)
            G = G_new

        self.G = G.detach().clone()
        return self

    @torch.no_grad()
    def project_language_matrix(self, lang: str, X: torch.Tensor) -> torch.Tensor:
        """
        Project a matrix of embeddings from one language.
        """
        return self.forward(lang, X)


# ============================================================
# 5) Collect multilingual language views
# ============================================================

@torch.no_grad()
def collect_language_views_from_loader(
    loader: DataLoader,
    embed_fn_by_lang: Dict[str, Callable[[List[str]], torch.Tensor]],
    device: str = "cpu",
    embed_batch_size: int = 64,
):
    """
    Builds one view per language with missing-view support.

    Returns:
        views[lang] = (X_lang, idx_lang)
        n_total = number of shared items

    Each TSV row defines one shared latent item id.
    Both src and tgt sentences for that row point to the same item id.
    """
    emb_lists = defaultdict(list)
    idx_lists = defaultdict(list)

    global_row = 0

    for src_langs, tgt_langs, s1, s2 in loader:
        B = len(s1)

        src_group = defaultdict(list)
        src_pos = defaultdict(list)
        for i, (lang, sent) in enumerate(zip(src_langs, s1)):
            src_group[lang].append(sent)
            src_pos[lang].append(global_row + i)

        tgt_group = defaultdict(list)
        tgt_pos = defaultdict(list)
        for i, (lang, sent) in enumerate(zip(tgt_langs, s2)):
            tgt_group[lang].append(sent)
            tgt_pos[lang].append(global_row + i)

        for lang, sents in src_group.items():
            if lang not in embed_fn_by_lang:
                raise KeyError(f"No embedder found for source language '{lang}'")
            E = embed_sentences_in_batches(
                sents,
                embed_fn_by_lang[lang],
                embed_batch_size,
                device,
            )
            emb_lists[lang].append(E)
            idx_lists[lang].append(torch.tensor(src_pos[lang], device=device))

        for lang, sents in tgt_group.items():
            if lang not in embed_fn_by_lang:
                raise KeyError(f"No embedder found for target language '{lang}'")
            E = embed_sentences_in_batches(
                sents,
                embed_fn_by_lang[lang],
                embed_batch_size,
                device,
            )
            emb_lists[lang].append(E)
            idx_lists[lang].append(torch.tensor(tgt_pos[lang], device=device))

        global_row += B

    views = {}
    for lang in emb_lists:
        X_lang = torch.cat(emb_lists[lang], dim=0)
        idx_lang = torch.cat(idx_lists[lang], dim=0)
        views[lang] = (X_lang, idx_lang)

    n_total = global_row
    return views, n_total


# ============================================================
# 7) End-to-end runner
# ============================================================

def run_gcca_training_example(
    tsv_path: str,
    seed: int,
    subset_size: int = 50000,
    d: int = 768,
    k: int = 256,
    epochs: int = 10,
    batch_size_pairs: int = 256,
    embed_batch_size: int = 64,
    use_dummy_embedder: bool = False,
    ollama_model: str = "llama3.1:8b",
    cache_dir: str = "./emb_cache_llama8b",
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
        tsv_path=tsv_path,
        cfg=cfg,
        batch_size_pairs=batch_size_pairs,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )

    if use_dummy_embedder:
        embed_fn_by_lang = None
    else:
        base = HFEmbedder(
            model_name=ollama_model,
            device=device,
            max_length=512,
            normalize=True,
        )
    cache = DiskEmbeddingCache(cache_dir)

    embed_fn_by_lang = {
        lang: CachedEmbedder(base, cache)
        for lang in languages
    }

    print(f"Languages: {languages}")
    print("Collecting training views...")
    views, n_total = collect_language_views_from_loader(
        loader=train_loader,
        embed_fn_by_lang=embed_fn_by_lang,
        device=device,
        embed_batch_size=embed_batch_size,
    )

    dims_by_lang = {lang: d for lang in languages}
    model = MaskedGCCAProjector(dims_by_lang=dims_by_lang, k=k).to(device)

    print("Fitting GCCA...")
    model.fit_gcca(
        views=views,
        n_total=n_total,
        epochs=epochs,
        reg=1e-4,
        verbose=True,
    )

    print("Evaluating on validation set...")

    return model


# ============================================================
# 8) Example main
# ============================================================

if __name__ == "__main__":
    # Replace with your file path
    TSV_PATH = "your_parallel_data.tsv"

    # Quick smoke test with dummy embeddings:
    # result = run_gcca_training_example(
    #     tsv_path=TSV_PATH,
    #     seed=42,
    #     subset_size=5000,
    #     d=768,
    #     k=256,
    #     epochs=5,
    #     batch_size_pairs=128,
    #     embed_batch_size=64,
    #     use_dummy_embedder=True,
    # )

    # Real run with Ollama:
    # result = run_gcca_training_example(
    #     tsv_path=TSV_PATH,
    #     seed=42,
    #     subset_size=50000,
    #     d=768,
    #     k=256,
    #     epochs=10,
    #     batch_size_pairs=128,
    #     embed_batch_size=32,
    #     use_dummy_embedder=False,
    #     ollama_model="llama3.1:8b",
    #     cache_dir="./emb_cache_llama8b",
    # )

    # print("Validation:", result["val_metrics"])
    # print("Test:", result["test_metrics"])
    pass