import torch
import torch.nn as nn


class PreNormConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.dilation = 1

        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(
            in_channels, out_channels, self.kernel_size, self.stride, self.padding
        )

    def calc_out_length(self, xlens: torch.Tensor) -> torch.Tensor:
        return (
            xlens + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
        ) // self.stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.relu(self.norm(x)))


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.pre_norm_conv1 = PreNormConv(
            in_channels, out_channels, kernel_size=3, stride=stride
        )
        self.pre_norm_conv2 = PreNormConv(
            out_channels, out_channels, kernel_size=3, stride=1
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_residual = x
        x = self.pre_norm_conv1(x)
        xlens = self.pre_norm_conv1.calc_out_length(xlens)

        x = self.pre_norm_conv2(x)
        xlens = self.pre_norm_conv2.calc_out_length(xlens)

        x = x + self.shortcut(x_residual)
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
