"""
A baseline using a randomly initialised ViT

Saves final model at experiments/random-init-vit/

A full working example can be found at:
https://www.kaggle.com/code/arnavnagpure/lejepa-exp-random-init-vit
"""
from lejepa.ViT import ViT
from pathlib import Path

import torch

from lejepa.config import TrainConfig
from lejepa.data import get_data
from lejepa.probe import linear_probe
from lejepa.utils import cache_embeddings
from lejepa.metrics import evaluate, effective_rank, mean_pairwise_cosine

out = Path(__file__).resolve().parent
config = TrainConfig(save_dir=str(out), epochs=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
ds = get_data(config.data_path, device)

image_size = ds['xtr'].shape[-1]
# Don't train
encoder = ViT(image_size, ds['mean'], ds['std'], config.model).to(device)
torch.save(encoder.state_dict(), out / "model.pt")


embeddings = cache_embeddings(encoder, ds["xtr"])
probe = linear_probe(embeddings, ds["ytr"], classes=10)
acc = evaluate(probe, cache_embeddings(encoder, ds["xte"]), ds["yte"], 4096)

print(
    f"test accuracy: {acc:.4f}, "
    f"erank: {effective_rank(embeddings):.4f}, "
    f"mp cos: {mean_pairwise_cosine(embeddings):.4f}"
)
