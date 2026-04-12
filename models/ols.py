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
    def fit_ols(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Fit projection using OLS.

        X : (n, d)
        Y : (n, k)
        """

        # compute W = (X^T X)^(-1) X^T Y
        XtX = X.T @ X
        XtY = X.T @ Y

        W = torch.linalg.solve(XtX, XtY)   # d x k

        # load into linear layer
        self.proj.weight.copy_(W.T)

def train_ols(data_loader, k, device="cpu", ridge=1e-4, d =768):
    first_batch = True
    XtX = None
    XtY = None
    inferred_d = None

    print("collecting sufficient statistics for OLS...")

    for _, e_i, e_j in data_loader:
        X = e_i.float().cpu()   # [b, d]
        Y = e_j.float().cpu()   # [b, d_y]

        if first_batch:
            inferred_d = X.shape[1]

            if Y.shape[1] != k:
                Y = Y[:, :k]

            XtX = torch.zeros(inferred_d, inferred_d, dtype=torch.float32)
            XtY = torch.zeros(inferred_d, k, dtype=torch.float32)
            first_batch = False
        else:
            if Y.shape[1] != k:
                Y = Y[:, :k]

        if X.shape[1] != inferred_d:
            raise ValueError(f"Inconsistent X dim: expected {inferred_d}, got {X.shape[1]}")

        XtX += X.T @ X
        XtY += X.T @ Y

    if XtX is None:
        raise ValueError("No batches were produced by data_loader.")

    model = OLS(inferred_d, k).to(device)

    I = torch.eye(inferred_d, dtype=torch.float32)
    W = torch.linalg.solve(XtX + ridge * I, XtY)

    model.proj.weight.data.copy_(W.T.to(device))
    return model

def run_ols_training_example(
    tsv_path: str,
    seed: int,
    subset_size: int = 50000,
    d: int = 768,
    k: int = 256,
    batches=16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ollama_model : str = "None", 
    cache_dir : str = "./ols_cache",
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
        pin_memory=(device.startswith("cuda")),
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
        neg_ratio=1.0
    )

    model = train_ols(train_batches, k=k, device=device)

    return model