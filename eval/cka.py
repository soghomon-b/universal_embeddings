import torch


@torch.no_grad()
def linear_cka_from_embeddings(Va: torch.Tensor, Vb: torch.Tensor, eps=1e-12):
    # Va: [n, da], Vb: [n, db]
    Va = Va - Va.mean(dim=0, keepdim=True)
    Vb = Vb - Vb.mean(dim=0, keepdim=True)

    cross = Va.T @ Vb  # [da, db]
    self_a = Va.T @ Va  # [da, da]
    self_b = Vb.T @ Vb  # [db, db]

    num = (cross**2).sum()  # ||Va^T Vb||_F^2
    den = torch.linalg.norm(self_a) * torch.linalg.norm(self_b) + eps

    return num / den
