from collections.abc import Iterator
import torch


def cache_embeddings(
    encoder: torch.nn.Module,
    x: torch.Tensor,
    batch_size: int = 1024,
) -> torch.Tensor:
    """
    Takes in a dataset and an encoder and encodes all items in the dataset
    once. Returns the cached dataset.

    Uses no_grad rather than inference_mode so the result is a normal tensor
    that a downstream probe can still backprop through.
    """
    encoder_was_training = encoder.training
    encoder.eval()

    with torch.no_grad():
        embeddings = [
            encoder(x[i : i + batch_size]) for i in range(0, len(x), batch_size)
        ]
        out = torch.cat(embeddings)

    if encoder_was_training:
        encoder.train()

    return out


def batch(
    n: int,
    batch_size: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    drop_last: bool = True,
) -> Iterator[torch.Tensor]:
    """
    A generator meant to bring the convenience of a DataLoader while
    being purely GPU bound (for smaller datasets)
    """
    if drop_last and batch_size > n:
        raise ValueError(
            f"batch_size {batch_size} exceeds dataset size {n} with drop_last=True, "
            f"which would yield no batches"
        )

    # Convert to device rather than using `device=` to
    # allow for CPU bound generators to work
    perm = torch.randperm(n, generator=generator).to(device)
    end = n - n % batch_size if drop_last else n

    for i in range(0, end, batch_size):
        yield perm[i : i + batch_size]
