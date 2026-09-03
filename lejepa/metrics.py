"""
Contains collapse metric helper functions to
help determine if representation collapse is occurring.
"""

import torch
import torch.nn.functional as F
from lejepa.utils import cache_embeddings


def entropy(p: torch.Tensor):
    """
    Calculates entropy from a probability distribution
    letting p*log(p) = 0 when p = 0
    """
    h = torch.where(p > 0, p * p.log(), 0.0)
    return -h.sum()


def effective_rank(z: torch.Tensor):
    """
    Calculates the effective rank of a distribution Z
    """
    # Make mean zero
    z = z - z.mean(dim=0)
    # Compute covariance matrix
    c = (z.T @ z) / len(z)
    # Calculate eigenvalues (=variances) for directions which maximise
    # covariance i.e. directions that are principle for distribution
    eigenvalues = torch.linalg.eigvalsh(c)

    # Check for total collapse (~one point)
    if eigenvalues.sum() < 1e-8:
        return torch.tensor(1.0, device=z.device)

    # Normalise to turn into a probability distribution
    p = eigenvalues / eigenvalues.sum()
    h = entropy(p)
    return torch.exp(h)  # Effective rank


def mean_pairwise_cosine(
    z: torch.Tensor, centered: bool = False, sample_size: int = 4096
):
    """
    Estimates the mean pairwise cosine of a distribution using only
    the absolute values of all the cosines
    """
    if centered:
        # Make mean zero / shift to origin
        z = z - z.mean(dim=0)

    # Take small sample for performance
    idx = torch.randperm(len(z))[:sample_size]
    z = z[idx]

    # Check for total collapse (~one point)
    norms = torch.linalg.vector_norm(z, dim=1)
    if (norms < 1e-16).all():
        return torch.tensor(1.0, device=z.device)

    # Normalise
    z = F.normalize(z, dim=1)
    cosine_pairs = z @ z.T
    n = len(cosine_pairs)

    # Sum all pairs and subtract the diagonal (which is always 1)
    # and divide by total number of remaining terms
    s = cosine_pairs.abs().sum() - n
    return s / (n**2 - n)


@torch.inference_mode()
def report(embedder: torch.nn.Module, x: torch.Tensor, label: str = "Stats"):
    """
    Prints a full report of a models stats including:
        effective rank,
        mean pairwise cosine (centered and uncentered)
        min/med/max of the stds
    """
    z = cache_embeddings(embedder, x)

    erank = effective_rank(z)
    mp_cos = mean_pairwise_cosine(z)
    mp_cos_cent = mean_pairwise_cosine(z, centered=True)
    std = z.std(dim=0)

    # Prints all in one line (broken here for readability)
    print(f"{label:3} \
    | erank {erank:.2f} \
    | cos {mp_cos:.3f} / {mp_cos_cent:.3f} \
    | std {std.min():.2f} / {std.median():.2f} / {std.max():.2f}")


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int
) -> float:
    """
    Calculates the models accuracy on dataset X with labels Y
    """
    assert len(x) == len(y)

    model_was_training = model.training
    model.eval()

    # Keep `correct` as GPU bound tensor to prevent
    # CPU sync every iteration
    correct = torch.zeros((), device=x.device)
    n = len(y)

    for i in range(0, n, batch_size):
        x_batch = x[i : i + batch_size]
        y_batch = y[i : i + batch_size]
        y_pred = model(x_batch)
        correct += (y_pred.argmax(1) == y_batch).sum()

    if model_was_training:
        model.train()

    return (correct / n).item()
