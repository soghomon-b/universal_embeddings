import torch
import torch.nn as nn
import torch.nn.functional as F
from .pair_wise import LinearProjector


def train_infonce(batches, d, k, lr=1e-3, epochs=5, tau=0.07, device="cpu"):
    model = LinearProjector(d, k).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"running Inforce training with {epochs}")
    for _ in range(epochs):
        for E1, E2 in batches:  # E1,E2: (N,d) where row i matches row i
            if not torch.is_tensor(E1):
                E1 = torch.tensor(E1, dtype=torch.float32)
            if not torch.is_tensor(E2):
                E2 = torch.tensor(E2, dtype=torch.float32)

            E1 = E1.to(device).float()  # (N,d)
            E2 = E2.to(device).float()  # (N,d)

            optimizer.zero_grad()

            z1 = model(E1)  # (N,k) normalized
            z2 = model(E2)  # (N,k) normalized

            logits = (z1 @ z2.T) / tau  # (N,N) similarities
            labels = torch.arange(z1.size(0), device=device)  # [0..N-1]

            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()

    return model


from .data_loader import (
    SplitConfig,
    prepare_parallel_data,
    OllamaEmbedder,
    DiskEmbeddingCache,
    CachedEmbedder,
    make_infonce_batches_from_loader,
)


def run_inforce_training_example(
    tsv_path: str,
    seed: int,
    subset_size: int = 50000,
    d: int = 768,
    k: int = 256,
    epochs=5,
    batches=256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
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
    embed_base = OllamaEmbedder(model="llama3.1:8b")
    cache = DiskEmbeddingCache("./emb_cache_granite278m")
    embed_src = CachedEmbedder(embed_base, cache)
    embed_tgt = embed_src  # same cache/model

    # Create an iterator/generator of (E1, E2) batches for InfoNCE training.
    train_batches = make_infonce_batches_from_loader(
        train_loader, embed_src, embed_tgt, device=device, embed_batch_size=batches
    )

    # Train projector (your function)
    model = train_infonce(train_batches, d=d, k=k, device=device, epochs=epochs)

    return model


if __name__ == "__main__":
    # Example usage:
    model = run_inforce_training_example(
        "", subset_size=100, d=10, k=11
    )
    print(model)
