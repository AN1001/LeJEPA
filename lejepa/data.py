"Contains functions to download and process the CIFAR-10 ds"

import os
import torch
from datasets import load_dataset, DatasetDict
import hashlib

# Expected hash value of correct downloaded data.
# Can be recomputed in data/compute_hash.py
# Verified on 1st Sep
EXPECTED_HASH = "9c70f2dd9c9023aed327b0df7d1b585077cccf552a673c09e540ce3d1d77b245"


def convert(
    ds: DatasetDict, split: str, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts CIFAR-10 HF ds to torch
    """
    # '[:]' to circumvent lazyness
    batch = ds[split].with_format("torch")[:]
    x = batch["img"].contiguous().to(device)
    y = batch["label"].to(device)

    return x, y


def build_ds(ds: DatasetDict, device: torch.device | str) -> dict[str, torch.Tensor]:
    """
    Builds the CIFAR-10 dataset, converting it to torch tensors and
    calculating the mean and std for the train split.

    Mean and std are for scaled data [0,1] since that
    is what gets used downsteam.
    """
    xtr, ytr = convert(ds, "train", device)
    xte, yte = convert(ds, "test", device)

    # Split train into train and val. e.g. 45k/5K
    data_perm = torch.randperm(50000, generator=torch.Generator().manual_seed(0))
    train_idx, val_idx = data_perm[:45000], data_perm[45000:]

    x_train = xtr[train_idx].float() / 255
    mean = x_train.mean((0, 2, 3), keepdim=True)
    std = x_train.std((0, 2, 3), keepdim=True)

    formatted_ds = {
        "xtr": xtr[train_idx],
        "ytr": ytr[train_idx],
        "xval": xtr[val_idx],
        "yval": ytr[val_idx],
        "xte": xte,
        "yte": yte,
        "mean": mean,
        "std": std,
    }

    return formatted_ds


def download_ds(save_path: str, device: torch.device | str) -> None:
    """
    Downloads CIFAR-10, converts it to torch tensors
    and saves it to memory
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Use HF to download since much faster than UofT
    ds = load_dataset("uoft-cs/cifar10")
    formatted_ds = build_ds(ds, device)

    validate_ds(formatted_ds)  # Validate before saving
    torch.save(formatted_ds, save_path)


def hash_ds(ds: dict[str, torch.Tensor]) -> str:
    """
    Calculate the SHA256 hash of the dataset ds
    """
    m = hashlib.sha256()
    for k in ("xtr", "ytr", "xval", "yval", "xte", "yte"):
        t = ds[k]
        m.update(f"{k}{tuple(t.shape)}{t.dtype}".encode())
        m.update(t.cpu().contiguous().numpy().tobytes())
    return m.hexdigest()


def validate_ds(ds: dict[str, torch.Tensor]) -> None:
    """
    Validates the dataset by hashing it and comparing
    its hash with a known correct dataset
    """
    actual_hash = hash_ds(ds)
    if actual_hash != EXPECTED_HASH:
        raise ValueError(f"""
            Data hash differed from expected;
            expected: {EXPECTED_HASH}
            got:      {actual_hash}
            """)


def get_data(path: str, device: torch.device | str) -> dict[str, torch.Tensor]:
    """
    Returns a dataset containing CIFAR-10 data.

    Returns as a dict containing keys:
        xtr, xte, xval, yval, ytr, yte, mean, std

    Downloads to path if not already saved there and validates
    the data is as expected.
    """
    if not os.path.isfile(path):
        download_ds(path, device)

    ds = torch.load(path, map_location=device, weights_only=True)
    validate_ds(ds)

    return ds
