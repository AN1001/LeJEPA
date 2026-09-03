from typing import Any
import torch
from lejepa.augment import build_view_transform
from lejepa.loss import loss_fn
from lejepa.data import get_data
from lejepa.ViT import ViT
from lejepa.config import TrainConfig
from lejepa.utils import batch, cache_embeddings
from lejepa.train_log import log_epoch
from lejepa.probe import linear_probe
from lejepa.metrics import evaluate
import torch.optim.lr_scheduler as scds


def train_epoch(
    embedder, optimiser, scheduler, transform, x_train, config, device
) -> tuple[torch.Tensor, torch.Tensor, int]:
    embedder.train()
    sim_total = torch.zeros((), device=device)
    sig_total = torch.zeros((), device=device)
    n_batches = 0

    for idx in batch(len(x_train), config.batch_size, device=device):
        x = x_train[idx]
        z_a = embedder(transform(x))
        z_b = embedder(transform(x))

        loss, sim, sig = loss_fn(z_a, z_b, config.lam, config.sigreg)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()

        sim_total += sim
        sig_total += sig
        n_batches += 1

    return sim_total, sig_total, n_batches


def probe_accuracy(embedder, ds, config: TrainConfig) -> float:
    """
    Fits a linear probe on the frozen encoder and scores it on the val split
    """
    embeddings = cache_embeddings(embedder, ds["xtr"])
    probe = linear_probe(embeddings, ds["ytr"], config.classes, config.probe)

    val_embeddings = cache_embeddings(embedder, ds["xval"])
    return evaluate(probe, val_embeddings, ds["yval"], config.probe.batch_size)


def build_optimiser(model, config, n_train):
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    warmup = scds.LinearLR(
        optimiser, start_factor=0.01, total_iters=config.warmup_steps
    )
    cosine = scds.CosineAnnealingLR(
        optimiser, config.total_steps(n_train) - config.warmup_steps
    )
    scheduler = scds.SequentialLR(
        optimiser, schedulers=[warmup, cosine], milestones=[config.warmup_steps]
    )
    return optimiser, scheduler


def train(
    config: TrainConfig = TrainConfig(), device: str | None = None
) -> torch.nn.Module:
    """
    Trains a model based on the configs specified. Collects its own
    data and returns the trained model
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)

    ds = get_data(config.data_path, device)
    transform = build_view_transform(config.augment)

    base = ViT(ds["xtr"].shape[-1], ds["mean"], ds["std"], config.model).to(device)
    embedder: Any = torch.compile(base)
    optimiser, scheduler = build_optimiser(base, config, len(ds["xtr"]))

    for epoch in range(1, config.epochs + 1):
        sim, sig, n_batches = train_epoch(
            embedder, optimiser, scheduler, transform, ds["xtr"], config, device
        )

        z = cache_embeddings(embedder, ds["xtr"][:8192])
        llr = scheduler.get_last_lr()[0]

        extras = {}
        if config.probe_every and epoch % config.probe_every == 0:
            extras["val_acc"] = probe_accuracy(embedder, ds, config)

        log_epoch(
            epoch, n_batches, sim.item(), sig.item(), z, config.save_dir, llr, extras,
        )

    return base
