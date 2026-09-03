"""
Full LeJEPA pretraining run: train + probe encoder

Logs and saves stats* every epoch at [TrainConfig.save_dir].
Also saves final model.

A full working example can be found at:
'you're already here!'


*Stats include:
similarity/sigreg loss, effective rank, mean pairwise cosine
(centered and uncentered), min/med/max of the embedding stds per dim
and learning rate
"""
from pathlib import Path

import torch

from lejepa.config import TrainConfig
from lejepa.data import get_data
from lejepa.probe import linear_probe
from lejepa.utils import cache_embeddings
from lejepa.metrics import evaluate, effective_rank, mean_pairwise_cosine
from lejepa.train import train

out = Path(__file__).resolve().parent
config = TrainConfig(save_dir=str(out))

encoder = train(config)
torch.save(encoder.state_dict(), out / "model.pt")

device = next(encoder.parameters()).device
ds = get_data(config.data_path, device)

embeddings = cache_embeddings(encoder, ds["xtr"])
probe = linear_probe(embeddings, ds["ytr"], classes=10)
acc = evaluate(probe, cache_embeddings(encoder, ds["xte"]), ds["yte"], 4096)

print(
    f"val accuracy: {acc:.4f}, "
    f"erank: {effective_rank(embeddings):.4f}, "
    f"mp cos: {mean_pairwise_cosine(embeddings):.4f}"
)
