from typing import List
import torch
import torch.nn as nn
from torch import Tensor
from .constant import *


class ResUnitBase(nn.Module):
    def __init__(self) -> None:
        super(ResUnitBase, self).__init__()

    def init_shortcut(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        bias: bool = True,
        padding_mode: str = PADDING_REPLICATE
    ) -> nn.Module:
        if out_channels != in_channels:
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=bias, padding_mode=padding_mode)
        else:
            return nn.Identity()


class ResUnit1(ResUnitBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: List[int] = [1, 1],
        padding: List[int] = [0, 1],
        dilation: int = 1,
        bias: bool = True,
        padding_mode: str = PADDING_REPLICATE,
        inplace: bool = True
    ) -> None:
        super(ResUnit1, self).__init__()

        self.weight_1: nn.Module = WeightLayer(
            in_channels, out_channels=in_channels, kernel_size=1, stride=stride[0], padding=padding[0],
            dilation=1, bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.weight_2: nn.Module = WeightLayer(
            in_channels, out_channels=in_channels, kernel_size=3, stride=stride[1], padding=padding[1],
            dilation=dilation, bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.weight_3: nn.Module = self.init_shortcut(
            in_channels, out_channels)
        self.shortcut: nn.Module = self.init_shortcut(
            in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        tx1: Tensor = self.weight_1(x)
        tx2: Tensor = self.weight_2(tx1)
        tx3: Tensor = self.weight_3(tx2)
        sc: Tensor = self.shortcut(x)
        return torch.add(tx3, sc)


class ResUnit2(ResUnitBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: List[int] = [1, 1],
        padding: List[int] = [0, 1],
        dilation: int = 1,
        bias: bool = True,
        padding_mode: str = PADDING_REPLICATE,
        inplace: bool = True
    ) -> None:
        super(ResUnit2, self).__init__()
        mid_channels: int = out_channels // 4

        self.relu: nn.Module = nn.ReLU(inplace)
        self.weight_1: nn.Module = WeightLayer(
            in_channels, out_channels=mid_channels, kernel_size=1, stride=stride[0], padding=padding[0],
            dilation=1, bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.weight_2: nn.Module = WeightLayer(
            in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=stride[1], padding=padding[1],
            dilation=dilation, bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.weight_3: nn.Module = self.init_shortcut(
            in_channels=mid_channels, out_channels=out_channels)

    def forward(self, x: Tensor) -> Tensor:
        tx1: Tensor = self.relu(x)
        tx2: Tensor = self.weight_1(tx1)
        tx3: Tensor = self.weight_2(tx2)
        tx4: Tensor = self.weight_3(tx3)
        return torch.add(tx4, x)


class ResUnit3(ResUnitBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: List[int] = [1, 2],
        padding: List[int] = [0, 1],
        dilation: int = 1,
        bias: bool = True,
        padding_mode: str = PADDING_REPLICATE,
        inplace: bool = True
    ) -> None:
        super(ResUnit3, self).__init__()
        mid_channels: int = in_channels // 2

        self.weight_1: nn.Module = WeightLayer(
            in_channels, out_channels=mid_channels, kernel_size=1, stride=stride[0], padding=padding[0],
            dilation=1, bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.weight_2: nn.Module = WeightLayer(
            in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=stride[1], padding=padding[1],
            dilation=dilation, bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.weight_3: nn.Module = self.init_shortcut(
            in_channels=mid_channels, out_channels=out_channels, stride=stride[0])
        self.shortcut: nn.Module = self.init_shortcut(
            in_channels, out_channels, stride=stride[1])

    def forward(self, x: Tensor) -> Tensor:
        tx1: Tensor = self.weight_1(x)
        tx2: Tensor = self.weight_2(tx1)
        tx3: Tensor = self.weight_3(tx2)
        sc: Tensor = self.shortcut(x)
        return torch.add(tx3, sc)


class ResUnit4(ResUnitBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: List[int] = [0, 1, 2],
        dilation: List[int] = [1, 1, 2],
        bias: bool = True,
        padding_mode: str = PADDING_REPLICATE,
        inplace: bool = True,
        half: bool = False
    ) -> None:
        super(ResUnit4, self).__init__()
        mid_channels_1: int = in_channels // 4
        mid_channels_2: int = in_channels // 16 if half is True else in_channels // 8
        mid_channels_3: int = in_channels // 8 if half is True else in_channels // 4

        self.bn: nn.Module = nn.BatchNorm2d(num_features=in_channels)
        self.relu: nn.Module = nn.ReLU(inplace)
        self.weight_1: nn.Module = WeightLayer(
            in_channels, out_channels=mid_channels_1, kernel_size=1, stride=stride, padding=padding[0],
            dilation=dilation[0], bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.conv_a: nn.Module = nn.Conv2d(
            in_channels=mid_channels_1, out_channels=mid_channels_2, kernel_size=3, stride=stride, padding=padding[1],
            dilation=dilation[1], bias=bias, padding_mode=padding_mode
        )
        self.conv_b: nn.Module = nn.Conv2d(
            in_channels=mid_channels_1, out_channels=mid_channels_2, kernel_size=3, stride=stride, padding=padding[2],
            dilation=dilation[2], bias=bias, padding_mode=padding_mode
        )
        self.weight_2: nn.Module = WeightLayer(
            in_channels=mid_channels_3, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding[0],
            dilation=dilation[0], bias=bias, padding_mode=padding_mode, inplace=inplace
        )

    def forward(self, x: Tensor) -> Tensor:
        tx1: Tensor = self.bn(x)
        tx2: Tensor = self.relu(tx1)
        tx3: Tensor = self.weight_1(tx2)
        tx4a: Tensor = self.conv_a(tx3)
        tx4b: Tensor = self.conv_b(tx3)
        tx5: Tensor = torch.cat((tx4a, tx4b), dim=1)
        tx6: Tensor = self.weight_2(tx5)
        return torch.add(tx6, x)


class ResUnit5(ResUnitBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: List[int] = [0, 2, 4],
        dilation: List[int] = [1, 2, 4],
        bias: bool = True,
        padding_mode: str = PADDING_REPLICATE,
        inplace: bool = True
    ) -> None:
        super(ResUnit5, self).__init__()
        mid_channels_1: int = in_channels // 2
        mid_channels_2: int = in_channels // 4

        self.weight_1: nn.Module = WeightLayer(
            in_channels, out_channels=mid_channels_1, kernel_size=1, stride=stride, padding=padding[0],
            dilation=dilation[0], bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.conv_a: nn.Module = nn.Conv2d(
            in_channels=mid_channels_1, out_channels=mid_channels_2, kernel_size=3, stride=stride, padding=padding[1],
            dilation=dilation[1], bias=bias, padding_mode=padding_mode
        )
        self.conv_b: nn.Module = nn.Conv2d(
            in_channels=mid_channels_1, out_channels=mid_channels_2, kernel_size=3, stride=stride, padding=padding[2],
            dilation=dilation[2], bias=bias, padding_mode=padding_mode
        )
        self.weight_2: nn.Module = WeightLayer(
            in_channels=mid_channels_1, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding[0],
            dilation=dilation[0], bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.shortcut: nn.Module = self.init_shortcut(
            in_channels, out_channels, bias=bias, padding_mode=padding_mode)

    def forward(self, x: Tensor) -> Tensor:
        tx1: Tensor = self.weight_1(x)
        tx2a: Tensor = self.conv_a(tx1)
        tx2b: Tensor = self.conv_b(tx1)
        tx3: Tensor = torch.cat((tx2a, tx2b), dim=1)
        tx4: Tensor = self.weight_2(tx3)
        sc: Tensor = self.shortcut(x)
        return torch.add(tx4, sc)


class WeightLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        padding_mode: str = PADDING_REPLICATE,
        inplace: bool = True,
        skip_bn: bool = False
    ):
        super().__init__()
        self.conv: nn.Module = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias, padding_mode=padding_mode)

        if skip_bn is True:
            self.bn: nn.Module = nn.Identity()
        else:
            self.bn: nn.Module = nn.BatchNorm2d(num_features=out_channels)

        self.relu: nn.Module = nn.ReLU(inplace)

    def forward(self, x: Tensor) -> torch.Tensor:
        tx1: Tensor = self.conv(x)
        tx2: Tensor = self.bn(tx1)
        tx3: Tensor = self.relu(tx2)
        return tx3


class AdapnetSkipLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        padding_mode: str = PADDING_REPLICATE
    ) -> None:
        super(AdapnetSkipLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                              padding=0, dilation=1, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x: Tensor) -> Tensor:
        tx1: Tensor = self.conv(x)
        return self.bn(tx1)


class TransposeWeightLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        padding_mode: str = PADDING_ZEROS
    ):
        super(TransposeWeightLayer, self).__init__()

        self.tconv: nn.Module = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias, dilation=dilation, padding_mode=padding_mode)
        self.bn: nn.Module = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x: Tensor) -> Tensor:
        tx: Tensor = self.tconv(x)
        return self.bn(tx)
