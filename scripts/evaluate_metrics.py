"""Simple helper script to check the metrics work in identifying known failures"""
from lejepa.metrics import effective_rank, mean_pairwise_cosine
import torch


rank_1         = torch.randn(4096, 1) @ torch.randn(1, 192)
rank_5         = torch.randn(4096, 5) @ torch.randn(5, 192)
rank_40        = torch.randn(4096, 40) @ torch.randn(40, 192)
point          = torch.randn(192).expand(4096, 192) * 3.0
offset         = torch.randn(4096, 192) + 10.0
scale_collapse = torch.randn(4096, 192) * 0.01

distributions = (rank_1, rank_5, rank_40, point, offset, scale_collapse)


for dist in distributions:
    print(f"Effective rank {effective_rank(dist):.4f} | MP cos {mean_pairwise_cosine(dist):.4f} | MP cos cent {mean_pairwise_cosine(dist):.4f}")
