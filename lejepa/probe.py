"""
Contains a function that creates, trains and tests a linear probe
to be fitted onto a frozen encoder.
"""

import torch
from lejepa.config import ProbeConfig
from lejepa.utils import batch


def _train_probe(
    probe: torch.nn.Module,
    embeddings: torch.Tensor,
    ytr: torch.Tensor,
    loss_fn: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    batch_size: int,
    epochs: int,
) -> None:
    probe.train()
    n = len(ytr)

    for _epoch in range(epochs):
        idxs = batch(n, batch_size, device=embeddings.device)

        for idx in idxs:
            batch_x, batch_y = embeddings[idx], ytr[idx]

            y_pred = probe(batch_x)
            loss = loss_fn(y_pred, batch_y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()


def linear_probe(
    embeddings: torch.Tensor,
    y: torch.Tensor,
    classes: int,
    config: ProbeConfig = ProbeConfig(),
) -> torch.nn.Module:
    """
    Creates and trains a linear probe to match embeddings to labels
    """
    device = y.device

    dims = embeddings.shape[1]

    probe = torch.nn.Linear(dims, classes).to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimiser = torch.optim.AdamW(probe.parameters(), lr=config.lr)

    _train_probe(
        probe, embeddings, y, loss_fn, optimiser, config.batch_size, config.epochs
    )

    return probe
