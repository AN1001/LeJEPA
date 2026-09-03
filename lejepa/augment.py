from torchvision.transforms import v2
from lejepa.config import AugmentConfig


def build_view_transform(config: AugmentConfig = AugmentConfig()) -> v2.Transform:
    return v2.Compose(
        [
            v2.RandomResizedCrop(size=(32, 32), scale=config.crop_scale),
            v2.RandomApply(
                [
                    v2.ColorJitter(
                        brightness=config.brightness,
                        contrast=config.contrast,
                        saturation=config.saturation,
                        hue=config.hue,
                    )
                ],
                p=config.jitter_p,
            ),
            v2.RandomGrayscale(p=config.greyscale_p),
            v2.RandomHorizontalFlip(p=config.flip_p)
        ]
    )
