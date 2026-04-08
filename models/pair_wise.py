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

class LinearProjector(nn.Module):
    def __init__(self, d: int, k: int):
        super().__init__()
        self.proj = nn.Linear(d, k, bias=False)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        z = self.proj(e)
        return F.normalize(z, dim=-1)


def train(data_loader, d, k, lr=1e-3, margin=0.0, epochs=5, device="cpu"):
    model = LinearProjector(d, k).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CosineEmbeddingLoss(margin=margin)
    batch = next(iter(data_loader))
    print(f"running pair_wise training with {epochs}")
    for _ in range(epochs):
        for y, e_i, e_j in data_loader:
            # e_i, e_j: (B, d)
            # y: (B,) with values +1 or -1

            e_i = e_i.to(device).float()
            e_j = e_j.to(device).float()
            y = y.to(device).float()

            optimizer.zero_grad()
            z_i = model(e_i)  # (B, k)
            z_j = model(e_j)  # (B, k)
            loss = criterion(z_i, z_j, y)
            loss.backward()
            optimizer.step()

    return model


def run_pairwise_training_example(
    tsv_path: str,
    seed: int,
    subset_size: int = 50000,
    d: int = 768,
    k: int = 256,
    epochs = 5,
    batches = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ollama_model : str = "None", 
    cache_dir : str = "./pairwise_cache",
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

    # Replace these with real multilingual sentence encoders.
    # For now, they’re dummy embedders to prove the pipeline works.
    embed_base = HFEmbedder(
        model_name=ollama_model,
        device=device)
    cache = DiskEmbeddingCache(cache_dir)
    embed_src = CachedEmbedder(embed_base, cache)
    embed_tgt = embed_src  # same cache/model

    # Create an iterator/generator of (E1, E2) batches for InfoNCE training.
    train_batches = make_pairwise_batches_from_loader(
    train_loader, embed_src, embed_tgt, device=device, embed_batch_size=batches, neg_ratio=1.0
)
    model = train(train_batches, d=d, k=k, device=device, epochs=epochs)

    return model


if __name__ == "__main__":
    # Example usage:
    model = run_pairwise_training_example(
        "", subset_size=100, d=10, k=11
    )
    print(model)
