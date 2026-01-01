import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1)
        self.dim_match = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1)),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_residual = x
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = x + self.dim_match(x_residual)
        return (x, xlens)


class Conv2DFeatureExtractor(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
    ) -> None:
        self.stem = nn.Conv2d(
            in_channels, out_channels=64, kernel_size=(3, 3), stride=1
        )
