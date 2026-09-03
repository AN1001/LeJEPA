from dataclasses import dataclass
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent / 'data'


@dataclass(frozen=True)
class ModelConfig:
    patch_size: int = 4
    dims: int       = 192
    heads: int      = 6
    blocks: int     = 5
    mlp_ratio: int  = 4
    # Scaled by 0.03 to ensure the position embeddings add information
    # whilst not drowning out the patch embedding
    pos_embed_ratio: float = 0.03

    def __post_init__(self):
        if self.dims % self.heads != 0:
            raise ValueError(f"Dims {self.dims} not divisible by heads {self.heads}")


@dataclass(frozen=True)
class SigregConfig:
    dirs: int        = 64
    grid_max: float  = 5.0
    grid_points: int = 17


@dataclass(frozen=True)
class ProbeConfig:
    lr: float = 1e-3
    batch_size: int = 1024
    epochs: int = 20


@dataclass(frozen=True)
class AugmentConfig:
    crop_scale: tuple[float, float] = (0.4, 1.0)
    brightness: tuple[float, float] = (0.4, 1.6)
    contrast: tuple[float, float]   = (0.4, 1.6)
    saturation: tuple[float, float] = (0.4, 1.6)
    hue: tuple[float, float]        = (-0.2, 0.2)
    jitter_p: float = 0.8
    greyscale_p: float = 0.2
    flip_p: float = 0.5


@dataclass(frozen=True)
class TrainConfig:
    seed: int      = 0
    lam: float     = 0.01
    epochs: int    = 600
    batch_size:int = 256
    lr: float      = 4e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000

    model: ModelConfig     = ModelConfig()
    sigreg: SigregConfig   = SigregConfig()
    augment: AugmentConfig = AugmentConfig()

    data_path: str = str(_DATA_DIR / 'cifar10.pt')
    save_dir: str = str(_DATA_DIR)

    def total_steps(self, n: int) -> int:
        return self.epochs * (n // self.batch_size)
