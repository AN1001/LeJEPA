import torch
from lejepa.ViT import Normaliser


class ConvBlock(torch.nn.Module):
    """
    Convolution -> Batch Norm -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm(self.conv(x))
        return self.relu(x)


class ConvLayer(torch.nn.Module):
    """
    ConvBlock -> ConvBlock -> Max Pool
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block_1 = ConvBlock(in_channels, out_channels)
        self.block_2 = ConvBlock(out_channels, out_channels)
        self.max_pool_1 = torch.nn.MaxPool2d(2)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        return self.max_pool_1(x)


class CNN(torch.nn.Module):
    """
    A supervised CNN with its own classification head.

    Input normalised and passed into a ConvLayer. Each ConvLayer
    halves the resolution and increases (e.g. doubles) the channel count.
    Finally global average pool and pass into a fully connected layer.

    [mean] and [std] are per channel and properties of the training data,
    both are in form (1, 3, 1, 1) for broadcasting.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        channels: tuple[int, int, int] = (32, 64, 128),
        classes: int = 10,
    ):
        super().__init__()
        self.input_norm = Normaliser(mean, std)

        self.conv_layers = torch.nn.Sequential()
        in_channels = 3
        for out_channels in channels:
            self.conv_layers.append(ConvLayer(in_channels, out_channels))
            in_channels = out_channels

        self.fc = torch.nn.Linear(in_channels, classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.conv_layers(x)
        x = x.mean(dim=(2, 3))
        return self.fc(x)
