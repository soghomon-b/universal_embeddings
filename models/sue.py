import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sklearn.cross_decomposition import CCA
except ImportError:
    CCA = None

try:
    from spectralnet import SpectralNet
except ImportError:
    SpectralNet = None

from .data_loader import (
    SplitConfig,
    prepare_parallel_data,
)


class SUE(nn.Module):
    def __init__(
        self,
        n_parallel: int = 100,
        n_components: int = 32,
        spectralnet_src_cfg: dict | None = None,
        spectralnet_tgt_cfg: dict | None = None,
        use_cca: bool = True,
        use_mmd: bool = False,
        normalize_output: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.n_parallel = n_parallel
        self.n_components = n_components
        self.use_cca = use_cca
        self.use_mmd = use_mmd
        self.normalize_output = normalize_output
        self.device = torch.device(device)

        if spectralnet_src_cfg is None:
            spectralnet_src_cfg = {}
        if spectralnet_tgt_cfg is None:
            spectralnet_tgt_cfg = {}

        spectralnet_src_cfg = dict(spectralnet_src_cfg)
        spectralnet_tgt_cfg = dict(spectralnet_tgt_cfg)

        src_k = spectralnet_src_cfg.get("n_clusters", n_components)
        tgt_k = spectralnet_tgt_cfg.get("n_clusters", n_components)

        spectralnet_src_cfg["n_clusters"] = src_k
        spectralnet_tgt_cfg["n_clusters"] = tgt_k

        src_hiddens = spectralnet_src_cfg.get("spectral_hiddens", [512, 512, src_k])
        tgt_hiddens = spectralnet_tgt_cfg.get("spectral_hiddens", [512, 512, tgt_k])

        src_hiddens = list(src_hiddens)
        tgt_hiddens = list(tgt_hiddens)

        src_hiddens[-1] = src_k
        tgt_hiddens[-1] = tgt_k

        spectralnet_src_cfg["spectral_hiddens"] = src_hiddens
        spectralnet_tgt_cfg["spectral_hiddens"] = tgt_hiddens

        self.spectralnet_src_cfg = spectralnet_src_cfg
        self.spectralnet_tgt_cfg = spectralnet_tgt_cfg


        self.src_model = None
        self.tgt_model = None
        self.cca = None
        self.projection_src = None
        self.projection_tgt = None
        self.mmd_model = None

        self.fitted_ = False

    def _check_inputs(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ):
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must both be 2D tensors of shape (n, d).")

        if X.size(0) != Y.size(0):
            raise ValueError("X and Y must have the same number of rows.")

        if X.size(0) == 0:
            raise ValueError("X and Y must be non-empty.")

        if self.n_parallel < 0 or self.n_parallel > X.size(0):
            raise ValueError("n_parallel must satisfy 0 <= n_parallel <= number of samples.")

        if SpectralNet is None:
            raise ImportError(
                "The 'spectralnet' package is required for SUE. "
                "Install the dependencies from the SUE setup first."
            )

        if self.use_cca and CCA is None:
            raise ImportError(
                "scikit-learn is required for CCA. Install it with: pip install scikit-learn"
            )

        if self.use_cca and self.n_parallel == 0:
            raise ValueError("CCA requires n_parallel > 0.")

    def _truncate_dims(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ):
        d = min(X.size(1), Y.size(1))
        return X[:, :d], Y[:, :d]

    def _to_numpy(
        self,
        X: torch.Tensor,
    ) -> np.ndarray:
        return X.detach().cpu().numpy()

    def _to_tensor(
        self,
        X: np.ndarray,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tensor(X, device=ref.device, dtype=ref.dtype)

    def _fit_spectral_models(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ):
        self.src_model = SpectralNet(**self.spectralnet_src_cfg)
        self.tgt_model = SpectralNet(**self.spectralnet_tgt_cfg)

        self.src_model.fit(X.float())
        self.tgt_model.fit(Y.float())

    def _spectral_transform_src(
        self,
        X: torch.Tensor,
    ) -> np.ndarray:
        self.src_model.transform(X.float())
        return self.src_model.embeddings_

    def _spectral_transform_tgt(
        self,
        Y: torch.Tensor,
    ) -> np.ndarray:
        self.tgt_model.transform(Y.float())
        return self.tgt_model.embeddings_

    def _fit_cca(
        self,
        Zx: np.ndarray,
        Zy: np.ndarray,
    ):
        if self.n_parallel == 0:
            self.projection_src = None
            self.projection_tgt = None
            return

        n_comp = min(
            self.n_components,
            Zx.shape[1],
            Zy.shape[1],
            self.n_parallel,
        )

        self.cca = CCA(n_components=n_comp)
        self.cca.fit(
            Zx[-self.n_parallel:],
            Zy[-self.n_parallel:],
        )

        self.projection_src = self.cca.x_rotations_
        self.projection_tgt = self.cca.y_rotations_

    def _apply_cca_src(
        self,
        Zx: np.ndarray,
    ) -> np.ndarray:
        if not self.use_cca or self.projection_src is None:
            return Zx
        return Zx @ self.projection_src

    def _apply_cca_tgt(
        self,
        Zy: np.ndarray,
    ) -> np.ndarray:
        if not self.use_cca or self.projection_tgt is None:
            return Zy
        return Zy @ self.projection_tgt

    def _maybe_normalize(
        self,
        Z: torch.Tensor,
    ) -> torch.Tensor:
        if self.normalize_output:
            Z = F.normalize(Z, dim=-1)
        return Z

    @torch.no_grad()
    def fit(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ):
        """
        X : (n, d_x)
        Y : (n, d_y)

        Assumes the last n_parallel rows are the paired anchor rows used for CCA.
        The rest can be treated as weakly paired / effectively unpaired for the
        spectral step.
        """

        self._check_inputs(X, Y)
        X, Y = self._truncate_dims(X, Y)

        X_cpu = X.detach().cpu()
        Y_cpu = Y.detach().cpu()

        self._fit_spectral_models(X_cpu, Y_cpu)

        Zx = self._spectral_transform_src(X_cpu)
        Zy = self._spectral_transform_tgt(Y_cpu)

        if self.use_cca:
            self._fit_cca(Zx, Zy)
            Zx = self._apply_cca_src(Zx)
            Zy = self._apply_cca_tgt(Zy)

        # Optional MMD hook:
        # The original SUE repo fine-tunes alignment with an MMD network after CCA.
        # I am leaving the hook here so you can plug in your own MMD module later
        # without changing the interface.
        if self.use_mmd:
            raise NotImplementedError(
                "use_mmd=True was requested, but no local MMD module is wired in yet. "
                "Keep use_mmd=False for now, or plug in your own MMD fine-tuning step."
            )

        Zx_t = self._to_tensor(Zx, X)
        Zy_t = self._to_tensor(Zy, Y)

        Zx_t = self._maybe_normalize(Zx_t)
        Zy_t = self._maybe_normalize(Zy_t)

        self.train_src_embeddings_ = Zx_t
        self.train_tgt_embeddings_ = Zy_t
        self.fitted_ = True

        return {
            "model": self,
            "src_embeddings": Zx_t,
            "tgt_embeddings": Zy_t,
        }

    @torch.no_grad()
    def transform_src(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform_src().")

        X_cpu = X.detach().cpu()
        Zx = self._spectral_transform_src(X_cpu)
        Zx = self._apply_cca_src(Zx)

        Zx_t = self._to_tensor(Zx, X)
        Zx_t = self._maybe_normalize(Zx_t)

        return Zx_t

    @torch.no_grad()
    def transform_tgt(
        self,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform_tgt().")

        Y_cpu = Y.detach().cpu()
        Zy = self._spectral_transform_tgt(Y_cpu)
        Zy = self._apply_cca_tgt(Zy)

        Zy_t = self._to_tensor(Zy, Y)
        Zy_t = self._maybe_normalize(Zy_t)

        return Zy_t

    @torch.no_grad()
    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ):
        """
        Returns:
            transformed_X, transformed_Y
        """
        return self.transform_src(X), self.transform_tgt(Y)


@torch.no_grad()
def fit_sue(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_parallel: int = 100,
    n_components: int = 32,
    spectralnet_src_cfg: dict | None = None,
    spectralnet_tgt_cfg: dict | None = None,
    use_cca: bool = True,
    use_mmd: bool = False,
    normalize_output: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = SUE(
        n_parallel=n_parallel,
        n_components=n_components,
        spectralnet_src_cfg=spectralnet_src_cfg,
        spectralnet_tgt_cfg=spectralnet_tgt_cfg,
        use_cca=use_cca,
        use_mmd=use_mmd,
        normalize_output=normalize_output,
        device=device,
    )

    result = model.fit(X, Y)
    return result


@torch.no_grad()
def apply_sue(
    model: SUE,
    X: torch.Tensor,
    Y: torch.Tensor,
):
    return model(X, Y)


@torch.no_grad()
def run_sue_example(
    tsv_path: str,
    seed: int,
    embedder,
    subset_size: int = 50000,
    batch_size_pairs: int = 64,
    n_parallel: int = 100,
    n_components: int = 32,
    spectralnet_src_cfg: dict | None = None,
    spectralnet_tgt_cfg: dict | None = None,
    use_cca: bool = True,
    use_mmd: bool = False,
    normalize_output: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
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

        X = embedder(src_texts).detach().cpu()
        Y = embedder(tgt_texts).detach().cpu()

        src_all.append(X)
        tgt_all.append(Y)

    X_train = torch.cat(src_all, dim=0)
    Y_train = torch.cat(tgt_all, dim=0)

    # Keep the final n_parallel rows as anchor pairs for the CCA step.
    # Since your TSV is paired, we can just use the last rows directly.
    result = fit_sue(
        X_train,
        Y_train,
        n_parallel=n_parallel,
        n_components=n_components,
        spectralnet_src_cfg=spectralnet_src_cfg,
        spectralnet_tgt_cfg=spectralnet_tgt_cfg,
        use_cca=use_cca,
        use_mmd=use_mmd,
        normalize_output=normalize_output,
        device=device,
    )

    return result