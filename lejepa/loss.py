import torch
from lejepa.sigreg import sigreg
from lejepa.config import SigregConfig


def loss_fn(
    z_a: torch.Tensor, z_b: torch.Tensor, lam: float,
    config: SigregConfig = SigregConfig()
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the loss:
        (1-λ)*sim + λ*sigreg

    z_a, z_b: embedded views
    """
    all_embeddings = torch.stack((z_a, z_b))

    centers = all_embeddings.mean(0)  # Find mean of every view pair

    # Each pair should encode the same information so find the difference/dist sq
    # and try to minimise it
    sim_loss = (centers.unsqueeze(0) - all_embeddings).square().mean()
    # Ensure the embeddings still stay well distributed (as an isotropic Gaussian)
    sig_loss = (sigreg(z_a, config) + sigreg(z_b, config)) * 0.5

    return (1 - lam) * sim_loss + lam * sig_loss, sim_loss.detach(), sig_loss.detach()
