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

        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)  # (b, t, 1)
        masked = token_embeddings * mask
        summed = masked.sum(dim=1)  # (b, h)
        counts = mask.sum(dim=1).clamp(min=1.0)  # (b, 1)

        return summed / counts


class BitextSentenceEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        max_length: int = 100,
    ):
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooler = MeanPooler()

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
        hidden = outputs.last_hidden_state  # (b, t, h)

        sent = self.pooler(hidden, batch["attention_mask"])  # (b, h)
        sent = F.normalize(sent, dim=-1)

        return sent

    def forward(self, src_texts, tgt_texts, device: str):
        z_src = self.encode_batch(src_texts, device=device)
        z_tgt = self.encode_batch(tgt_texts, device=device)
        return z_src, z_tgt


def symmetric_bitext_loss(
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    temperature: float = 0.05,
):
    """
    z_src : (b, h), already normalized
    z_tgt : (b, h), already normalized

    We use in-batch negatives:
      - for each source, all other targets in batch are negatives
      - for each target, all other sources in batch are negatives
    """

    logits = (z_src @ z_tgt.T) / temperature  # (b, b)
    targets = torch.arange(z_src.size(0), device=z_src.device)

    loss_src_to_tgt = F.cross_entropy(logits, targets)
    loss_tgt_to_src = F.cross_entropy(logits.T, targets)

    loss = 0.5 * (loss_src_to_tgt + loss_tgt_to_src)

    return loss, logits


@torch.no_grad()
def retrieval_accuracy(logits: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    gold = torch.arange(logits.size(0), device=logits.device)
    return (preds == gold).float().mean().item()


@torch.no_grad()
def evaluate_bitext(
    model: BitextSentenceEncoder,
    data_loader,
    device: str = "cpu",
    temperature: float = 0.05,
):
    model.eval()

    total_loss = 0.0
    total_acc_src_to_tgt = 0.0
    total_acc_tgt_to_src = 0.0
    total_batches = 0

    for batch in data_loader:
        _, src_texts, tgt_texts = batch

        z_src, z_tgt = model(src_texts, tgt_texts, device=device)
        loss, logits = symmetric_bitext_loss(
            z_src,
            z_tgt,
            temperature=temperature,
        )

        acc_src_to_tgt = retrieval_accuracy(logits)
        acc_tgt_to_src = retrieval_accuracy(logits.T)

        total_loss += loss.item()
        total_acc_src_to_tgt += acc_src_to_tgt
        total_acc_tgt_to_src += acc_tgt_to_src
        total_batches += 1

    if total_batches == 0:
        return {
            "loss": math.nan,
            "acc_src_to_tgt": math.nan,
            "acc_tgt_to_src": math.nan,
        }

    return {
        "loss": total_loss / total_batches,
        "acc_src_to_tgt": total_acc_src_to_tgt / total_batches,
        "acc_tgt_to_src": total_acc_tgt_to_src / total_batches,
    }


def train_bitext_encoder(
    train_loader,
    val_loader,
    model_name: str = "xlm-roberta-base",
    max_length: int = 100,
    lr: float = 2e-5,
    weight_decay: float = 1e-2,
    epochs: int = 1,
    temperature: float = 0.05,
    device: str = "cpu",
):
    model = BitextSentenceEncoder(
        model_name=model_name,
        max_length=max_length,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    best_state = None

    print("starting bitext training...")
    print(f"model = {model_name}")
    print(f"device = {device}")

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_acc_src_to_tgt = 0.0
        total_acc_tgt_to_src = 0.0
        total_batches = 0
        first_batch = next(iter(train_loader))
        print(type(first_batch))
        if isinstance(first_batch, (list, tuple)):
            print("len =", len(first_batch))
            for i, x in enumerate(first_batch):
                print(i, type(x), x[:2] if hasattr(x, "__getitem__") else x)
        elif isinstance(first_batch, dict):
            print(first_batch.keys())

        for batch_idx, batch in enumerate(train_loader):
            _, src_texts, tgt_texts = batch

            optimizer.zero_grad()

            z_src, z_tgt = model(src_texts, tgt_texts, device=device)
            loss, logits = symmetric_bitext_loss(
                z_src,
                z_tgt,
                temperature=temperature,
            )

            loss.backward()
            optimizer.step()

            acc_src_to_tgt = retrieval_accuracy(logits)
            acc_tgt_to_src = retrieval_accuracy(logits.T)

            total_loss += loss.item()
            total_acc_src_to_tgt += acc_src_to_tgt
            total_acc_tgt_to_src += acc_tgt_to_src
            total_batches += 1

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"epoch {epoch + 1} | batch {batch_idx + 1} | "
                    f"loss = {total_loss / total_batches:.4f} | "
                    f"src->tgt acc = {total_acc_src_to_tgt / total_batches:.4f} | "
                    f"tgt->src acc = {total_acc_tgt_to_src / total_batches:.4f}"
                )

        train_metrics = {
            "loss": total_loss / max(total_batches, 1),
            "acc_src_to_tgt": total_acc_src_to_tgt / max(total_batches, 1),
            "acc_tgt_to_src": total_acc_tgt_to_src / max(total_batches, 1),
        }

        val_metrics = evaluate_bitext(
            model,
            val_loader,
            device=device,
            temperature=temperature,
        )

        print(
            f"[epoch {epoch + 1}/{epochs}] "
            f"train_loss = {train_metrics['loss']:.4f}, "
            f"train_src->tgt = {train_metrics['acc_src_to_tgt']:.4f}, "
            f"train_tgt->src = {train_metrics['acc_tgt_to_src']:.4f}, "
            f"val_loss = {val_metrics['loss']:.4f}, "
            f"val_src->tgt = {val_metrics['acc_src_to_tgt']:.4f}, "
            f"val_tgt->src = {val_metrics['acc_tgt_to_src']:.4f}"
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
            }

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    return model


@torch.no_grad()
def encode_texts(
    model: BitextSentenceEncoder,
    texts,
    device: str = "cpu",
    batch_size: int = 64,
):
    model.eval()

    all_vecs = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        z = model.encode_batch(chunk, device=device)
        all_vecs.append(z.cpu())

    return torch.cat(all_vecs, dim=0)


def run_bitext_training_example(
    tsv_path: str,
    seed: int,
    subset_size: int = 50000,
    model_name: str = "xlm-roberta-base",
    max_length: int = 100,
    batch_size_pairs: int = 64,
    lr: float = 2e-5,
    weight_decay: float = 1e-2,
    epochs: int = 1,
    temperature: float = 0.05,
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

    model = train_bitext_encoder(
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=model_name,
        max_length=max_length,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        temperature=temperature,
        device=device,
    )

    return model