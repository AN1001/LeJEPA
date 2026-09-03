"""
Downloads a fresh dataset from HF and computes its hash for
use when validating downloads in data.py
"""

from lejepa.data import build_ds, hash_ds
from datasets import load_dataset

ds = load_dataset("uoft-cs/cifar10")
formatted_ds = build_ds(ds, "cpu")

h = hash_ds(formatted_ds)
print(f"Computed hash:\n{h}")
