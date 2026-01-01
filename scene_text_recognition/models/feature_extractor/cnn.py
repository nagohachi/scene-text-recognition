from typing import Literal

import torch
import torch.nn as nn

from scene_text_recognition.models.feature_extractor.schemas import FeatureExtractorBase

Stride = int | tuple[int, int]


def _to_tuple(stride: Stride) -> tuple[int, int]:
    if isinstance(stride, int):
        return (stride, stride)
    return stride


class PreNormConv(FeatureExtractorBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Stride = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = _to_tuple(stride)
        self.padding = kernel_size // 2
        self.dilation = 1

        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(
            in_channels, out_channels, self.kernel_size, self.stride, self.padding
        )

    def calc_out_length(self, xlens: torch.Tensor) -> torch.Tensor:
        stride_w = self.stride[1]
        return (
            xlens + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
        ) // stride_w + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.relu(self.norm(x)))


class BasicBlockV2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: Stride) -> None:
        super().__init__()
        stride_tuple = _to_tuple(stride)
        self.pre_norm_conv1 = PreNormConv(
            in_channels, out_channels, kernel_size=3, stride=stride
        )
        self.pre_norm_conv2 = PreNormConv(
            out_channels, out_channels, kernel_size=3, stride=1
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride_tuple
                ),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels or stride_tuple != (1, 1)
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


class ResnetV2Stage(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: Stride, num_blocks: int
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            if i == 0:
                self.blocks.append(BasicBlockV2(in_channels, out_channels, stride))
            else:
                self.blocks.append(BasicBlockV2(out_channels, out_channels, 1))

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x, xlens = block(x, xlens)
        return x, xlens


class ResnetV2FeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, type: Literal["resnet18",]) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        match type:
            case "resnet18":
                self.stages = nn.ModuleList(
                    [
                        ResnetV2Stage(64, 64, (2, 2), 2),
                        ResnetV2Stage(64, 128, (2, 2), 2),
                        ResnetV2Stage(128, 256, (2, 1), 2),
                        ResnetV2Stage(256, 512, (2, 1), 2),
                    ]
                )
            case _:
                raise NotImplementedError()

    @property
    def out_size(self) -> int:
        return 512

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        for stage in self.stages:
            x, xlens = stage(x, xlens)
        return x, xlens
