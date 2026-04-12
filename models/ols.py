import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_loader import (
    SplitConfig,
    prepare_parallel_data,
    HFEmbedder,
    DiskEmbeddingCache,
    CachedEmbedder,
    make_pairwise_batches_from_loader,
)


class OLS(nn.Module):
    def __init__(self, d: int, k: int):
        super().__init__()
        self.proj = nn.Linear(d, k, bias=False)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        z = self.proj(e)
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def fit_from_stats(self, XtX: torch.Tensor, XtY: torch.Tensor, ridge: float = 1e-4):
        """
        Fit projection using sufficient statistics.

        XtX : (d, d)
        XtY : (d, k)
        """
        d = XtX.shape[0]
        I = torch.eye(d, dtype=XtX.dtype, device=XtX.device)
        W = torch.linalg.solve(XtX + ridge * I, XtY)   # (d, k)
        self.proj.weight.copy_(W.T)


def train_ols(data_loader, k: int, device: str = "cpu", ridge: float = 1e-4):
    """
    Train OLS by streaming sufficient statistics:
        XtX = sum(X^T X)
        XtY = sum(X^T Y)

    This avoids storing all batches in memory.
    """
    XtX = None
    XtY = None
    inferred_d = None
    num_batches = 0
    num_examples = 0

    print("collecting sufficient statistics for OLS...")

    for _, e_i, e_j in data_loader:
        X = e_i.float().cpu()   # (b, d_in)
        Y = e_j.float().cpu()   # (b, d_tgt)

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError(f"Expected 2D tensors, got X.shape={X.shape}, Y.shape={Y.shape}")

        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Batch size mismatch: X.shape={X.shape}, Y.shape={Y.shape}")

        if inferred_d is None:
            inferred_d = X.shape[1]
            XtX = torch.zeros(inferred_d, inferred_d, dtype=torch.float32)
            XtY = torch.zeros(inferred_d, k, dtype=torch.float32)
            print(f"[OLS] detected input dim d={inferred_d}, target dim={Y.shape[1]}, proj dim k={k}")

        if X.shape[1] != inferred_d:
            raise ValueError(f"Inconsistent input dim: expected {inferred_d}, got {X.shape[1]}")

        # If target dim is larger than k, truncate.
        # If target dim is smaller than k, fail loudly.
        if Y.shape[1] < k:
            raise ValueError(f"Target dim {Y.shape[1]} is smaller than k={k}")

        Yk = Y[:, :k]

        XtX += X.T @ X
        XtY += X.T @ Yk

        num_batches += 1
        num_examples += X.shape[0]

    if XtX is None or XtY is None:
        raise ValueError("No batches were produced by data_loader.")

    print(f"[OLS] accumulated {num_examples} examples across {num_batches} batches")

    model = OLS(inferred_d, k).to(device)
    model.fit_from_stats(XtX, XtY, ridge=ridge)

    # parameters must live on target device
    model.proj.weight.data.copy_(model.proj.weight.data.to(device))
    return model


def run_ols_training_example(
    tsv_path: str,
    seed: int,
    subset_size: int = 50000,
    k: int = 256,
    batches: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ollama_model: str = "None",
    cache_dir: str = "./ols_cache",
    ridge: float = 1e-4,
):
    cfg = SplitConfig(
        subset_size=subset_size,
        train_frac=0.90,
        val_frac=0.05,
        test_frac=0.05,
        seed=seed,
    )

    train_loader, val_loader, test_loader = prepare_parallel_data(
        tsv_path,
        cfg,
        batch_size_pairs=16,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )

    embed_base = HFEmbedder(
        model_name=ollama_model,
        device=device,
    )

    cache = DiskEmbeddingCache(cache_dir)
    embed_src = CachedEmbedder(embed_base, cache)
    embed_tgt = embed_src

    train_batches = make_pairwise_batches_from_loader(
        train_loader,
        embed_src,
        embed_tgt,
        device=device,
        embed_batch_size=batches,
        neg_ratio=1.0,
    )

    model = train_ols(
        train_batches,
        k=k,
        device=device,
        ridge=ridge,
    )

    return model