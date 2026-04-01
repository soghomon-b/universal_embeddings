import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .data_loader import (
    SplitConfig,
    prepare_parallel_data,
)


class MeanPooler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        token_embeddings : (b, t, h)
        attention_mask   : (b, t)
        returns          : (b, h)
        """
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        masked = token_embeddings * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts


class BitextSentenceBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        max_length: int = 100,
        normalize_outputs: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length
        self.normalize_outputs = normalize_outputs

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooler = MeanPooler()
        self.hidden_size = self.encoder.config.hidden_size

    def tokenize(self, texts):
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def encode_batch(self, texts, device: str):
        batch = self.tokenize(texts)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = self.encoder(**batch)
        hidden = outputs.last_hidden_state
        sent = self.pooler(hidden, batch["attention_mask"])

        if self.normalize_outputs:
            sent = F.normalize(sent, dim=-1)

        return sent


class GaussianMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-10.0, max=10.0)
        return mu, logvar


class MLPDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor):
        return self.net(z)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor):
    """
    returns scalar batch-mean KL(q(z|x) || N(0, I))
    """
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    kl_per_ex = kl_per_dim.sum(dim=-1)
    return kl_per_ex.mean()


def product_of_gaussians(
    mu_a: torch.Tensor,
    logvar_a: torch.Tensor,
    mu_b: torch.Tensor,
    logvar_b: torch.Tensor,
    eps: float = 1e-8,
):
    """
    Combine two diagonal Gaussian posteriors.

    returns:
      joint_mu, joint_logvar
    """
    var_a = torch.exp(logvar_a).clamp(min=eps)
    var_b = torch.exp(logvar_b).clamp(min=eps)

    prec_a = 1.0 / var_a
    prec_b = 1.0 / var_b

    joint_var = 1.0 / (prec_a + prec_b)
    joint_mu = joint_var * (mu_a * prec_a + mu_b * prec_b)
    joint_logvar = torch.log(joint_var.clamp(min=eps))

    return joint_mu, joint_logvar


class DVCCA(nn.Module):
    """
    Practical DVCCA-style model for paired sentence embeddings.

    Shared latent:
      q_s(z_s | x_src)
      q_s(z_s | x_tgt)

    Private latents:
      q_p_src(z_p_src | x_src)
      q_p_tgt(z_p_tgt | x_tgt)

    Joint shared posterior during training is formed by product-of-experts
    over the two shared posteriors.
    """

    def __init__(
        self,
        input_dim: int,
        shared_dim: int = 256,
        private_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.shared_dim = shared_dim
        self.private_dim = private_dim

        self.shared_src_encoder = GaussianMLP(
            in_dim=input_dim,
            latent_dim=shared_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.shared_tgt_encoder = GaussianMLP(
            in_dim=input_dim,
            latent_dim=shared_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.private_src_encoder = GaussianMLP(
            in_dim=input_dim,
            latent_dim=private_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.private_tgt_encoder = GaussianMLP(
            in_dim=input_dim,
            latent_dim=private_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.src_decoder = MLPDecoder(
            in_dim=shared_dim + private_dim,
            out_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.tgt_decoder = MLPDecoder(
            in_dim=shared_dim + private_dim,
            out_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x_src: torch.Tensor, x_tgt: torch.Tensor):
        mu_s_src, logvar_s_src = self.shared_src_encoder(x_src)
        mu_s_tgt, logvar_s_tgt = self.shared_tgt_encoder(x_tgt)

        mu_p_src, logvar_p_src = self.private_src_encoder(x_src)
        mu_p_tgt, logvar_p_tgt = self.private_tgt_encoder(x_tgt)

        mu_s_joint, logvar_s_joint = product_of_gaussians(
            mu_s_src,
            logvar_s_src,
            mu_s_tgt,
            logvar_s_tgt,
        )

        z_s = reparameterize(mu_s_joint, logvar_s_joint)
        z_p_src = reparameterize(mu_p_src, logvar_p_src)
        z_p_tgt = reparameterize(mu_p_tgt, logvar_p_tgt)

        x_src_hat = self.src_decoder(torch.cat([z_s, z_p_src], dim=-1))
        x_tgt_hat = self.tgt_decoder(torch.cat([z_s, z_p_tgt], dim=-1))

        return {
            "x_src_hat": x_src_hat,
            "x_tgt_hat": x_tgt_hat,
            "mu_s_src": mu_s_src,
            "logvar_s_src": logvar_s_src,
            "mu_s_tgt": mu_s_tgt,
            "logvar_s_tgt": logvar_s_tgt,
            "mu_s_joint": mu_s_joint,
            "logvar_s_joint": logvar_s_joint,
            "mu_p_src": mu_p_src,
            "logvar_p_src": logvar_p_src,
            "mu_p_tgt": mu_p_tgt,
            "logvar_p_tgt": logvar_p_tgt,
        }

    @torch.no_grad()
    def encode_shared_src(self, x_src: torch.Tensor):
        mu, _ = self.shared_src_encoder(x_src)
        return F.normalize(mu, dim=-1)

    @torch.no_grad()
    def encode_shared_tgt(self, x_tgt: torch.Tensor):
        mu, _ = self.shared_tgt_encoder(x_tgt)
        return F.normalize(mu, dim=-1)


class BitextDVCCA(nn.Module):
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        max_length: int = 100,
        shared_dim: int = 256,
        private_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.backbone = BitextSentenceBackbone(
            model_name=model_name,
            max_length=max_length,
            normalize_outputs=True,
        )

        self.dvcca = DVCCA(
            input_dim=self.backbone.hidden_size,
            shared_dim=shared_dim,
            private_dim=private_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def encode_inputs(self, src_texts, tgt_texts, device: str):
        if self.freeze_backbone:
            with torch.no_grad():
                x_src = self.backbone.encode_batch(src_texts, device=device)
                x_tgt = self.backbone.encode_batch(tgt_texts, device=device)
        else:
            x_src = self.backbone.encode_batch(src_texts, device=device)
            x_tgt = self.backbone.encode_batch(tgt_texts, device=device)

        return x_src, x_tgt

    def forward(self, src_texts, tgt_texts, device: str):
        x_src, x_tgt = self.encode_inputs(src_texts, tgt_texts, device=device)
        outputs = self.dvcca(x_src, x_tgt)
        outputs["x_src"] = x_src
        outputs["x_tgt"] = x_tgt
        return outputs

    @torch.no_grad()
    def encode_shared_texts(
        self,
        texts,
        device: str,
        view: str = "src",
    ):
        x = self.backbone.encode_batch(texts, device=device)

        if view == "src":
            z = self.dvcca.encode_shared_src(x)
        elif view == "tgt":
            z = self.dvcca.encode_shared_tgt(x)
        else:
            raise ValueError("view must be one of {'src', 'tgt'}")

        return z


def dvcca_loss(
    outputs,
    recon_weight: float = 1.0,
    kl_shared_weight: float = 1.0,
    kl_private_weight: float = 1.0,
    align_weight: float = 1.0,
):
    x_src = outputs["x_src"]
    x_tgt = outputs["x_tgt"]
    x_src_hat = outputs["x_src_hat"]
    x_tgt_hat = outputs["x_tgt_hat"]

    mu_s_src = outputs["mu_s_src"]
    logvar_s_src = outputs["logvar_s_src"]
    mu_s_tgt = outputs["mu_s_tgt"]
    logvar_s_tgt = outputs["logvar_s_tgt"]

    mu_s_joint = outputs["mu_s_joint"]
    logvar_s_joint = outputs["logvar_s_joint"]

    mu_p_src = outputs["mu_p_src"]
    logvar_p_src = outputs["logvar_p_src"]
    mu_p_tgt = outputs["mu_p_tgt"]
    logvar_p_tgt = outputs["logvar_p_tgt"]

    recon_src = F.mse_loss(x_src_hat, x_src, reduction="mean")
    recon_tgt = F.mse_loss(x_tgt_hat, x_tgt, reduction="mean")
    recon = recon_src + recon_tgt

    kl_shared_joint = kl_standard_normal(mu_s_joint, logvar_s_joint)
    kl_shared_src = kl_standard_normal(mu_s_src, logvar_s_src)
    kl_shared_tgt = kl_standard_normal(mu_s_tgt, logvar_s_tgt)
    kl_shared = kl_shared_joint + 0.5 * (kl_shared_src + kl_shared_tgt)

    kl_private_src = kl_standard_normal(mu_p_src, logvar_p_src)
    kl_private_tgt = kl_standard_normal(mu_p_tgt, logvar_p_tgt)
    kl_private = kl_private_src + kl_private_tgt

    align = 1.0 - F.cosine_similarity(mu_s_src, mu_s_tgt, dim=-1).mean()

    loss = (
        recon_weight * recon
        + kl_shared_weight * kl_shared
        + kl_private_weight * kl_private
        + align_weight * align
    )

    return {
        "loss": loss,
        "recon": recon.detach(),
        "recon_src": recon_src.detach(),
        "recon_tgt": recon_tgt.detach(),
        "kl_shared": kl_shared.detach(),
        "kl_private": kl_private.detach(),
        "align": align.detach(),
    }


@torch.no_grad()
def shared_retrieval_accuracy(
    mu_src: torch.Tensor,
    mu_tgt: torch.Tensor,
) -> float:
    mu_src = F.normalize(mu_src, dim=-1)
    mu_tgt = F.normalize(mu_tgt, dim=-1)

    sims = mu_src @ mu_tgt.T
    preds = sims.argmax(dim=1)
    gold = torch.arange(sims.size(0), device=sims.device)
    return (preds == gold).float().mean().item()


@torch.no_grad()
def evaluate_dvcca(
    model: BitextDVCCA,
    data_loader,
    device: str = "cpu",
    recon_weight: float = 1.0,
    kl_shared_weight: float = 1.0,
    kl_private_weight: float = 1.0,
    align_weight: float = 1.0,
):
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl_shared = 0.0
    total_kl_private = 0.0
    total_align = 0.0
    total_acc_src_to_tgt = 0.0
    total_acc_tgt_to_src = 0.0
    total_batches = 0

    for batch in data_loader:
        _, _, _, src_texts, tgt_texts = batch

        outputs = model(src_texts, tgt_texts, device=device)
        losses = dvcca_loss(
            outputs,
            recon_weight=recon_weight,
            kl_shared_weight=kl_shared_weight,
            kl_private_weight=kl_private_weight,
            align_weight=align_weight,
        )

        acc_src_to_tgt = shared_retrieval_accuracy(
            outputs["mu_s_src"],
            outputs["mu_s_tgt"],
        )
        acc_tgt_to_src = shared_retrieval_accuracy(
            outputs["mu_s_tgt"],
            outputs["mu_s_src"],
        )

        total_loss += losses["loss"].item()
        total_recon += losses["recon"].item()
        total_kl_shared += losses["kl_shared"].item()
        total_kl_private += losses["kl_private"].item()
        total_align += losses["align"].item()
        total_acc_src_to_tgt += acc_src_to_tgt
        total_acc_tgt_to_src += acc_tgt_to_src
        total_batches += 1

    if total_batches == 0:
        return {
            "loss": math.nan,
            "recon": math.nan,
            "kl_shared": math.nan,
            "kl_private": math.nan,
            "align": math.nan,
            "acc_src_to_tgt": math.nan,
            "acc_tgt_to_src": math.nan,
        }

    return {
        "loss": total_loss / total_batches,
        "recon": total_recon / total_batches,
        "kl_shared": total_kl_shared / total_batches,
        "kl_private": total_kl_private / total_batches,
        "align": total_align / total_batches,
        "acc_src_to_tgt": total_acc_src_to_tgt / total_batches,
        "acc_tgt_to_src": total_acc_tgt_to_src / total_batches,
    }


def train_dvcca(
    train_loader,
    val_loader,
    model_name: str = "xlm-roberta-base",
    max_length: int = 100,
    shared_dim: int = 256,
    private_dim: int = 128,
    hidden_dim: int = 512,
    dropout: float = 0.1,
    freeze_backbone: bool = True,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 5,
    recon_weight: float = 1.0,
    kl_shared_weight: float = 1e-3,
    kl_private_weight: float = 1e-3,
    align_weight: float = 1.0,
    device: str = "cpu",
):
    model = BitextDVCCA(
        model_name=model_name,
        max_length=max_length,
        shared_dim=shared_dim,
        private_dim=private_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    best_state = None

    print("starting DVCCA training...")
    print(f"model = {model_name}")
    print(f"device = {device}")
    print(f"freeze_backbone = {freeze_backbone}")
    print(f"shared_dim = {shared_dim}, private_dim = {private_dim}")

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_kl_shared = 0.0
        total_kl_private = 0.0
        total_align = 0.0
        total_acc_src_to_tgt = 0.0
        total_acc_tgt_to_src = 0.0
        total_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            _, _, src_texts, tgt_texts = batch

            optimizer.zero_grad()

            outputs = model(src_texts, tgt_texts, device=device)
            losses = dvcca_loss(
                outputs,
                recon_weight=recon_weight,
                kl_shared_weight=kl_shared_weight,
                kl_private_weight=kl_private_weight,
                align_weight=align_weight,
            )

            losses["loss"].backward()
            optimizer.step()

            acc_src_to_tgt = shared_retrieval_accuracy(
                outputs["mu_s_src"].detach(),
                outputs["mu_s_tgt"].detach(),
            )
            acc_tgt_to_src = shared_retrieval_accuracy(
                outputs["mu_s_tgt"].detach(),
                outputs["mu_s_src"].detach(),
            )

            total_loss += losses["loss"].item()
            total_recon += losses["recon"].item()
            total_kl_shared += losses["kl_shared"].item()
            total_kl_private += losses["kl_private"].item()
            total_align += losses["align"].item()
            total_acc_src_to_tgt += acc_src_to_tgt
            total_acc_tgt_to_src += acc_tgt_to_src
            total_batches += 1

        val_metrics = evaluate_dvcca(
            model,
            val_loader,
            device=device,
            recon_weight=recon_weight,
            kl_shared_weight=kl_shared_weight,
            kl_private_weight=kl_private_weight,
            align_weight=align_weight,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {
                "model_state_dict": {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                },
                "model_name": model_name,
                "max_length": max_length,
                "shared_dim": shared_dim,
                "private_dim": private_dim,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "freeze_backbone": freeze_backbone,
            }

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    return model


@torch.no_grad()
def encode_texts_dvcca(
    model: BitextDVCCA,
    texts,
    device: str = "cpu",
    batch_size: int = 64,
    view: str = "src",
):
    model.eval()

    all_vecs = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        z = model.encode_shared_texts(
            chunk,
            device=device,
            view=view,
        )
        all_vecs.append(z.cpu())

    return torch.cat(all_vecs, dim=0)


def run_dvcca_training_example(
    tsv_path: str,
    seed: int,
    subset_size: int = 50000,
    model_name: str = "xlm-roberta-base",
    max_length: int = 100,
    batch_size_pairs: int = 64,
    shared_dim: int = 256,
    private_dim: int = 128,
    hidden_dim: int = 512,
    dropout: float = 0.1,
    freeze_backbone: bool = True,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 5,
    recon_weight: float = 1.0,
    kl_shared_weight: float = 1e-3,
    kl_private_weight: float = 1e-3,
    align_weight: float = 1.0,
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
        batch_size_pairs=batch_size_pairs,
        num_workers=0,
        pin_memory=(device.startswith("cuda")),
    )

    model = train_dvcca(
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=model_name,
        max_length=max_length,
        shared_dim=shared_dim,
        private_dim=private_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        recon_weight=recon_weight,
        kl_shared_weight=kl_shared_weight,
        kl_private_weight=kl_private_weight,
        align_weight=align_weight,
        device=device,
    )

    return model


import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Assumes these already exist earlier in your file:
# - BitextSentenceBackbone
# - DVCCA
# - dvcca_loss
# - shared_retrieval_accuracy
# - SplitConfig
# - prepare_parallel_data


class BitextDVCCA(nn.Module):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 64,
        shared_dim: int = 128,
        private_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.backbone = BitextSentenceBackbone(
            model_name=model_name,
            max_length=max_length,
            normalize_outputs=True,
        )

        self.dvcca = DVCCA(
            input_dim=self.backbone.hidden_size,
            shared_dim=shared_dim,
            private_dim=private_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def encode_inputs(self, src_texts, tgt_texts, device: str):
        if self.freeze_backbone:
            with torch.no_grad():
                x_src = self.backbone.encode_batch(src_texts, device=device)
                x_tgt = self.backbone.encode_batch(tgt_texts, device=device)
        else:
            x_src = self.backbone.encode_batch(src_texts, device=device)
            x_tgt = self.backbone.encode_batch(tgt_texts, device=device)

        return x_src, x_tgt

    def forward(self, src_texts, tgt_texts, device: str):
        x_src, x_tgt = self.encode_inputs(src_texts, tgt_texts, device=device)
        outputs = self.dvcca(x_src, x_tgt)
        outputs["x_src"] = x_src
        outputs["x_tgt"] = x_tgt
        return outputs

    @torch.no_grad()
    def encode_shared_texts(
        self,
        texts,
        device: str,
        view: str = "src",
    ):
        self.eval()
        self.backbone.eval()
        self.dvcca.eval()

        x = self.backbone.encode_batch(texts, device=device)

        if view == "src":
            z = self.dvcca.encode_shared_src(x)
        elif view == "tgt":
            z = self.dvcca.encode_shared_tgt(x)
        else:
            raise ValueError("view must be one of {'src', 'tgt'}")

        return z


class PairedEmbeddingDataset(Dataset):
    def __init__(self, x_src: torch.Tensor, x_tgt: torch.Tensor):
        if x_src.ndim != 2 or x_tgt.ndim != 2:
            raise ValueError("x_src and x_tgt must both be 2D tensors.")
        if x_src.size(0) != x_tgt.size(0):
            raise ValueError("x_src and x_tgt must have the same number of rows.")

        self.x_src = x_src.contiguous()
        self.x_tgt = x_tgt.contiguous()

    def __len__(self):
        return self.x_src.size(0)

    def __getitem__(self, idx):
        return self.x_src[idx], self.x_tgt[idx]


class CachedBitextDVCCA(nn.Module):
    """
    Wrapper used after cached training:
    - backbone is used only for inference-time text -> embedding
    - dvcca is trained only on cached backbone embeddings
    """

    def __init__(self, backbone: BitextSentenceBackbone, dvcca: DVCCA):
        super().__init__()
        self.backbone = backbone
        self.dvcca = dvcca

    @torch.no_grad()
    def encode_shared_texts(
        self,
        texts,
        device: str,
        view: str = "src",
    ):
        self.eval()
        self.backbone.eval().to(device)
        self.dvcca.eval().to(device)

        x = self.backbone.encode_batch(texts, device=device)

        if view == "src":
            z = self.dvcca.encode_shared_src(x)
        elif view == "tgt":
            z = self.dvcca.encode_shared_tgt(x)
        else:
            raise ValueError("view must be one of {'src', 'tgt'}")

        return z


@torch.no_grad()
def cache_backbone_embeddings(
    backbone: BitextSentenceBackbone,
    data_loader,
    device: str = "cpu",
):
    """
    Expects batches shaped like:
        (src_langs, tgt_langs, src_texts, tgt_texts)
    """
    backbone.eval().to(device)

    src_all = []
    tgt_all = []

    for batch in data_loader:
        _, _, src_texts, tgt_texts = batch

        x_src = backbone.encode_batch(src_texts, device=device)
        x_tgt = backbone.encode_batch(tgt_texts, device=device)

        src_all.append(x_src.detach().cpu())
        tgt_all.append(x_tgt.detach().cpu())

    if not src_all:
        raise ValueError("data_loader produced no batches.")

    return torch.cat(src_all, dim=0), torch.cat(tgt_all, dim=0)


@torch.no_grad()
def evaluate_dvcca_cached(
    model: DVCCA,
    data_loader,
    device: str = "cpu",
    recon_weight: float = 1.0,
    kl_shared_weight: float = 1.0,
    kl_private_weight: float = 1.0,
    align_weight: float = 1.0,
):
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl_shared = 0.0
    total_kl_private = 0.0
    total_align = 0.0
    total_acc_src_to_tgt = 0.0
    total_acc_tgt_to_src = 0.0
    total_batches = 0

    for x_src, x_tgt in data_loader:
        x_src = x_src.to(device, non_blocking=True)
        x_tgt = x_tgt.to(device, non_blocking=True)

        outputs = model(x_src, x_tgt)
        outputs["x_src"] = x_src
        outputs["x_tgt"] = x_tgt

        losses = dvcca_loss(
            outputs,
            recon_weight=recon_weight,
            kl_shared_weight=kl_shared_weight,
            kl_private_weight=kl_private_weight,
            align_weight=align_weight,
        )

        acc_src_to_tgt = shared_retrieval_accuracy(
            outputs["mu_s_src"],
            outputs["mu_s_tgt"],
        )
        acc_tgt_to_src = shared_retrieval_accuracy(
            outputs["mu_s_tgt"],
            outputs["mu_s_src"],
        )

        total_loss += losses["loss"].item()
        total_recon += losses["recon"].item()
        total_kl_shared += losses["kl_shared"].item()
        total_kl_private += losses["kl_private"].item()
        total_align += losses["align"].item()
        total_acc_src_to_tgt += acc_src_to_tgt
        total_acc_tgt_to_src += acc_tgt_to_src
        total_batches += 1

    if total_batches == 0:
        return {
            "loss": math.nan,
            "recon": math.nan,
            "kl_shared": math.nan,
            "kl_private": math.nan,
            "align": math.nan,
            "acc_src_to_tgt": math.nan,
            "acc_tgt_to_src": math.nan,
        }

    return {
        "loss": total_loss / total_batches,
        "recon": total_recon / total_batches,
        "kl_shared": total_kl_shared / total_batches,
        "kl_private": total_kl_private / total_batches,
        "align": total_align / total_batches,
        "acc_src_to_tgt": total_acc_src_to_tgt / total_batches,
        "acc_tgt_to_src": total_acc_tgt_to_src / total_batches,
    }


def train_dvcca_cached(
    train_loader,
    val_loader,
    input_dim: int,
    shared_dim: int = 128,
    private_dim: int = 64,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 5,
    recon_weight: float = 1.0,
    kl_shared_weight: float = 1e-3,
    kl_private_weight: float = 1e-3,
    align_weight: float = 1.0,
    device: str = "cpu",
    use_amp: bool = True,
):
    model = DVCCA(
        input_dim=input_dim,
        shared_dim=shared_dim,
        private_dim=private_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    use_cuda_amp = use_amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    best_val_loss = float("inf")
    best_state = None

    print("starting cached DVCCA training.")
    print(f"device = {device}")
    print(f"input_dim = {input_dim}")
    print(f"shared_dim = {shared_dim}, private_dim = {private_dim}, hidden_dim = {hidden_dim}")
    print(f"use_amp = {use_cuda_amp}")

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_kl_shared = 0.0
        total_kl_private = 0.0
        total_align = 0.0
        total_acc_src_to_tgt = 0.0
        total_acc_tgt_to_src = 0.0
        total_batches = 0

        for x_src, x_tgt in train_loader:
            x_src = x_src.to(device, non_blocking=True)
            x_tgt = x_tgt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                outputs = model(x_src, x_tgt)
                outputs["x_src"] = x_src
                outputs["x_tgt"] = x_tgt

                losses = dvcca_loss(
                    outputs,
                    recon_weight=recon_weight,
                    kl_shared_weight=kl_shared_weight,
                    kl_private_weight=kl_private_weight,
                    align_weight=align_weight,
                )

            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()

            acc_src_to_tgt = shared_retrieval_accuracy(
                outputs["mu_s_src"].detach(),
                outputs["mu_s_tgt"].detach(),
            )
            acc_tgt_to_src = shared_retrieval_accuracy(
                outputs["mu_s_tgt"].detach(),
                outputs["mu_s_src"].detach(),
            )

            total_loss += losses["loss"].item()
            total_recon += losses["recon"].item()
            total_kl_shared += losses["kl_shared"].item()
            total_kl_private += losses["kl_private"].item()
            total_align += losses["align"].item()
            total_acc_src_to_tgt += acc_src_to_tgt
            total_acc_tgt_to_src += acc_tgt_to_src
            total_batches += 1

        train_metrics = {
            "loss": total_loss / max(total_batches, 1),
            "recon": total_recon / max(total_batches, 1),
            "kl_shared": total_kl_shared / max(total_batches, 1),
            "kl_private": total_kl_private / max(total_batches, 1),
            "align": total_align / max(total_batches, 1),
            "acc_src_to_tgt": total_acc_src_to_tgt / max(total_batches, 1),
            "acc_tgt_to_src": total_acc_tgt_to_src / max(total_batches, 1),
        }

        val_metrics = evaluate_dvcca_cached(
            model,
            val_loader,
            device=device,
            recon_weight=recon_weight,
            kl_shared_weight=kl_shared_weight,
            kl_private_weight=kl_private_weight,
            align_weight=align_weight,
        )

        print(
            f"[epoch {epoch + 1}] "
            f"train loss = {train_metrics['loss']:.4f} | "
            f"val loss = {val_metrics['loss']:.4f} | "
            f"train src->tgt = {train_metrics['acc_src_to_tgt']:.4f} | "
            f"train tgt->src = {train_metrics['acc_tgt_to_src']:.4f} | "
            f"val src->tgt = {val_metrics['acc_src_to_tgt']:.4f} | "
            f"val tgt->src = {val_metrics['acc_tgt_to_src']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


@torch.no_grad()
def encode_texts_dvcca(
    model,
    texts,
    device: str = "cpu",
    batch_size: int = 64,
    view: str = "src",
):
    model.eval()

    all_vecs = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        z = model.encode_shared_texts(
            chunk,
            device=device,
            view=view,
        )
        all_vecs.append(z.cpu())

    return torch.cat(all_vecs, dim=0)


def run_dvcca_training_example(
    tsv_path: str,
    seed: int,
    subset_size: int = 50000,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_length: int = 64,
    batch_size_pairs: int = 128,
    shared_dim: int = 128,
    private_dim: int = 64,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 5,
    recon_weight: float = 1.0,
    kl_shared_weight: float = 1e-3,
    kl_private_weight: float = 1e-3,
    align_weight: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    cache_train: bool = True,
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
        batch_size_pairs=batch_size_pairs,
        num_workers=2,
        pin_memory=device.startswith("cuda"),
    )

    if not cache_train:
        # Fallback to original-style training if you ever want it again
        model = BitextDVCCA(
            model_name=model_name,
            max_length=max_length,
            shared_dim=shared_dim,
            private_dim=private_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            freeze_backbone=True,
        ).to(device)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        best_val_loss = float("inf")
        best_state = None
        use_cuda_amp = device.startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

        print("starting uncached DVCCA training.")
        print(f"model = {model_name}")
        print(f"device = {device}")

        for epoch in range(epochs):
            model.train()

            for batch in train_loader:
                _, _, src_texts, tgt_texts = batch

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                    outputs = model(src_texts, tgt_texts, device=device)
                    losses = dvcca_loss(
                        outputs,
                        recon_weight=recon_weight,
                        kl_shared_weight=kl_shared_weight,
                        kl_private_weight=kl_private_weight,
                        align_weight=align_weight,
                    )

                scaler.scale(losses["loss"]).backward()
                scaler.step(optimizer)
                scaler.update()

            val_metrics = evaluate_dvcca(
                model,
                val_loader,
                device=device,
                recon_weight=recon_weight,
                kl_shared_weight=kl_shared_weight,
                kl_private_weight=kl_private_weight,
                align_weight=align_weight,
            )

            print(f"[epoch {epoch + 1}] uncached val loss = {val_metrics['loss']:.4f}")

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

        if best_state is not None:
            model.load_state_dict(best_state)

        return model

    # Fast path: cache backbone embeddings once, then train only DVCCA
    backbone = BitextSentenceBackbone(
        model_name=model_name,
        max_length=max_length,
        normalize_outputs=True,
    )

    print("caching train backbone embeddings...")
    x_src_train, x_tgt_train = cache_backbone_embeddings(
        backbone=backbone,
        data_loader=train_loader,
        device=device,
    )

    print("caching val backbone embeddings...")
    x_src_val, x_tgt_val = cache_backbone_embeddings(
        backbone=backbone,
        data_loader=val_loader,
        device=device,
    )

    train_cached_loader = DataLoader(
        PairedEmbeddingDataset(x_src_train, x_tgt_train),
        batch_size=batch_size_pairs,
        shuffle=True,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )

    val_cached_loader = DataLoader(
        PairedEmbeddingDataset(x_src_val, x_tgt_val),
        batch_size=batch_size_pairs,
        shuffle=False,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )

    dvcca = train_dvcca_cached(
        train_loader=train_cached_loader,
        val_loader=val_cached_loader,
        input_dim=backbone.hidden_size,
        shared_dim=shared_dim,
        private_dim=private_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        recon_weight=recon_weight,
        kl_shared_weight=kl_shared_weight,
        kl_private_weight=kl_private_weight,
        align_weight=align_weight,
        device=device,
        use_amp=True,
    )

    return CachedBitextDVCCA(backbone, dvcca)