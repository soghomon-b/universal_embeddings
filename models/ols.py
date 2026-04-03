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

def train_ols(data_loader, d, k, device="cpu"):
    model = OLS(d, k).to(device)

    X_all = []
    Y_all = []

    print("collecting embeddings for OLS...")

    for y, e_i, e_j in data_loader:
        e_i = e_i.to(device).float()
        e_j = e_j.to(device).float()

        X_all.append(e_i)
        Y_all.append(e_j)

    X = torch.cat(X_all, dim=0)  # (n, d)
    Y = torch.cat(Y_all, dim=0)  # (n, d)

    # If target dim differs
    if Y.shape[1] != k:
        Y = Y[:, :k]

    model.fit_ols(X, Y)

    return model

def run_ols_training_example(
    tsv_path: str,
    seed: int,
    subset_size: int = 50000,
    d: int = 768,
    k: int = 256,
    batches=256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ollama_model : str = "None"
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
        batch_size_pairs=256,
        num_workers=0,
        pin_memory=(device.startswith("cuda")),
    )

    embed_base = HFEmbedder(
            model_name=ollama_model,
            device=device,
        )
    cache = DiskEmbeddingCache("./emb_cache_llama8b")
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

    model = train_ols(train_batches, d=d, k=k, device=device)

    return model