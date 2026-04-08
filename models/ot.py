import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import ot
except ImportError:
    ot = None

from .data_loader import (
    SplitConfig,
    prepare_parallel_data,
)


class SinkhornOT(nn.Module):
    def __init__(
        self,
        reg: float = 0.1,
        metric: str = "cosine",
        normalize_inputs: bool = True,
    ):
        super().__init__()

        self.reg = reg
        self.metric = metric
        self.normalize_inputs = normalize_inputs

    def _check_inputs(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ):
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must both be 2D tensors of shape (n, d).")

        if X.size(1) != Y.size(1):
            raise ValueError("X and Y must have the same embedding dimension.")

        if X.size(0) == 0 or Y.size(0) == 0:
            raise ValueError("X and Y must be non-empty.")

        if ot is None:
            raise ImportError(
                "The 'pot' package is required for Optimal Transport. "
                "Install it with: pip install POT"
            )

    def _prepare_inputs(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ):
        if self.normalize_inputs:
            X = F.normalize(X, dim=-1)
            Y = F.normalize(Y, dim=-1)

        return X, Y

    def _cost_matrix(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        if self.metric == "cosine":
            return 1.0 - (X @ Y.T)

        if self.metric == "euclidean":
            return torch.cdist(X, Y, p=2)

        raise ValueError(f"Unsupported metric: {self.metric}")

    def _uniform_marginals(
        self,
        n: int,
        m: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        a = torch.full((n,), 1.0 / n, device=device, dtype=dtype)
        b = torch.full((m,), 1.0 / m, device=device, dtype=dtype)
        return a, b

    @torch.no_grad()
    def transport_plan(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        """
        X : (n, d)
        Y : (m, d)

        returns T : (n, m)
        """

        self._check_inputs(X, Y)

        X, Y = self._prepare_inputs(X, Y)

        C = self._cost_matrix(X, Y)
        a, b = self._uniform_marginals(
            X.size(0),
            Y.size(0),
            device=X.device,
            dtype=X.dtype,
        )

        T = ot.sinkhorn(
            a.detach().cpu().numpy(),
            b.detach().cpu().numpy(),
            C.detach().cpu().numpy(),
            reg=self.reg,
        )

        T = torch.tensor(T, device=X.device, dtype=X.dtype)

        return T

    @torch.no_grad()
    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        """
        X : (n, d)
        Y : (m, d)

        returns transported_X : (n, d)

        transported_X[i] = sum_j T[i, j] * Y[j]
        """

        T = self.transport_plan(X, Y)
        transported_X = T @ Y

        if self.normalize_inputs:
            transported_X = F.normalize(transported_X, dim=-1)

        return transported_X


@torch.no_grad()
def fit_sinkhorn_ot(
    X: torch.Tensor,
    Y: torch.Tensor,
    reg: float = 0.1,
    metric: str = "cosine",
    normalize_inputs: bool = True,
):
    model = SinkhornOT(
        reg=reg,
        metric=metric,
        normalize_inputs=normalize_inputs,
    )

    transported_X = model(X, Y)

    return {
        "model": model,
        "transported_X": transported_X,
    }


@torch.no_grad()
def apply_sinkhorn_ot(
    model: SinkhornOT,
    X: torch.Tensor,
    Y: torch.Tensor,
):
    return model(X, Y)


def run_sinkhorn_ot_example(
    tsv_path: str,
    seed: int,
    embedder,
    subset_size: int = 50000,
    batch_size_pairs: int = 64,
    reg: float = 0.1,
    metric: str = "cosine",
    normalize_inputs: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    cache_dir : str = "./ot_cache",
):
    cfg = SplitConfig(
        subset_size=subset_size,
        train_frac=0.90,
        val_frac=0.05,
        test_frac=0.05,
        seed=seed,
    )

    train_loader, _, _ = prepare_parallel_data(
        tsv_path,
        cfg,
        batch_size_pairs=batch_size_pairs,
        num_workers=0,
        pin_memory=(device.startswith("cuda")),
    )

    src_all = []
    tgt_all = []

    
    for batch in train_loader:
        _, _, src_texts, tgt_texts = batch

        X = embedder(src_texts).to(device)
        Y = embedder(tgt_texts).to(device)

        src_all.append(X)
        tgt_all.append(Y)

    X_train = torch.cat(src_all, dim=0)
    Y_train = torch.cat(tgt_all, dim=0)

    result = fit_sinkhorn_ot(
        X_train,
        Y_train,
        reg=reg,
        metric=metric,
        normalize_inputs=normalize_inputs,
    )

    return result