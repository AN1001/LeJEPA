"""
Records per epoch training stats to a JSON log and prints them.
"""

import json
import os
import torch
from lejepa.metrics import effective_rank, mean_pairwise_cosine


def _update_log(path: str, row: dict):
    if os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
    else:
        data = []

    data.append(row)

    with open(path, 'w') as f:
        json.dump(data, f)


def _print_log(row):
    out = (
        f"E {row['epoch']:<4}"
        f"| erank {row['erank']:>8.4f} "
        f"| cos {row['cos']:.4f} / {row['cos_cent']:.4f} "
        f"| lr {row['lr']:.2e} "
        f"| sig {row['sig_loss']:>9.4f} "
        f"| sim {row['sim_loss']:>7.4f}"
    )

    print(out)


def log_epoch(
    epoch: int,
    n_b: int,
    sim_total: float,
    sig_total: float,
    z: torch.Tensor,
    save_dir: str,
    lr: float
):
    row = {
        'epoch': epoch,
        'sim_loss': sim_total / n_b,
        'sig_loss': sig_total / n_b,
        'erank': effective_rank(z).item(),
        'cos': mean_pairwise_cosine(z).item(),
        'cos_cent': mean_pairwise_cosine(z, True).item(),
        'lr': lr
    }
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'train_log.json')
    _update_log(path, row)
    _print_log(row)
