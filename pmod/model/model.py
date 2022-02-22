from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn as nn

from .module import *
from .constant import *

__all__ = ['PMOD', 'Discriminator', 'UnetDiscriminator']


class PMOD(nn.Module):
    def __init__(self, camera_shape: Tuple[int, int, int], num_classes: int):
        """PMOD

        Args:
            camera_shape (Tuple[int, int, int]): (C, H, W)
            num_classes (int): Classes of Semantic Segmentation
        """
        super().__init__()
        assert isinstance(
            num_classes, int) and num_classes > 1, '"num_classes" must be an integer greater than 1.'

        in_shape: Tuple[int, int, int] = (
            camera_shape[0] + 1, *camera_shape[1:])

        self.encoder = AdapnetEncoder(in_shape)
        encoder_out_shape: List[Tuple[int, int, int]
                                ] = self.encoder.get_out_shape()

        self.easpp = eAspp(encoder_out_shape[-1])
        encoder_out_shape.append(self.easpp.get_out_shape())

        self.decoder_seg = AdapnetDecoder(
            in_shapes=encoder_out_shape, out_channels=num_classes, use_aux=True)
        self.decoder_depth = AdapnetDecoder(
            in_shapes=encoder_out_shape, out_channels=1, use_aux=False, b3_channels=32)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, camera: Tensor, map_depth: Tensor) -> PMOD_OUTPUT:
        input_tensor: Tensor = torch.cat((camera, map_depth), dim=1)
        skips: List[Tensor] = self.encoder(input_tensor)

        skips.append(self.easpp(skips[-1]))

        segmentation: ADAPNET_DECODER_OUTPUT = self.decoder_seg(skips)
        depth: ADAPNET_DECODER_OUTPUT = self.decoder_depth(skips)

        return PMOD_OUTPUT(
            segmentation.output, self.relu(depth.output),
            segmentation.aux1, segmentation.aux2
        )

    def forward_adapnet(self, camera: Tensor, map_depth: Tensor) -> PMOD_OUTPUT:
        input_tensor: Tensor = torch.cat((camera, map_depth), dim=1)
        skips: List[Tensor] = self.encoder(input_tensor)

        skips.append(self.easpp(skips[-1]))

        segmentation: ADAPNET_DECODER_OUTPUT = self.decoder_seg(skips)
        depth: ADAPNET_DECODER_OUTPUT = self.decoder_depth(skips)

        return PMOD_OUTPUT(
            segmentation.output, self.relu(depth.output),
            segmentation.aux1, segmentation.aux2
        )


class AdapnetEncoder(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        bias: bool = True,
        padding_mode: str = PADDING_REPLICATE,
        inplace: bool = True
    ):
        """__init__

        Args:
            in_shape (Tuple[int, int, int]): (C, H, W)
            bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
            padding_mode (bool, str): 'zeros', 'reflect', 'replicate' or 'circular'. Defaults to 'replicate'.
            inplace (bool, optional): can optionally do the operation in-place. Defaults to True.
        """
        super(AdapnetEncoder, self).__init__()

        channels, height, width = in_shape

        self._out_shapes: List[Tuple[int, int, int]] = []

        # Block 1
        b1_channels: int = BASE_CHANNELS * 4
        self.b1_1_bn: nn.Module = nn.BatchNorm2d(num_features=channels)
        self.b1_2_wl: nn.Module = WeightLayer(
            in_channels=channels, out_channels=BASE_CHANNELS, kernel_size=7, stride=2, padding=3, bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.b1_3_mp: nn.Module = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b1_4_u1: nn.Module = ResUnit1(
            in_channels=BASE_CHANNELS, out_channels=b1_channels, bias=bias, padding_mode=padding_mode, inplace=inplace)
        self.b1_5_u2: nn.Module = ResUnit2(
            in_channels=b1_channels, out_channels=b1_channels, bias=bias, padding_mode=padding_mode, inplace=inplace)
        self.b1_6_u2: nn.Module = ResUnit2(
            in_channels=b1_channels, out_channels=b1_channels, bias=bias, padding_mode=padding_mode, inplace=inplace)
        self.b1_7_re: nn.Module = nn.ReLU(inplace)
        self._out_shapes.append((b1_channels, height // 4, width // 4))

        # Block 2
        b2_channels: int = b1_channels * 2
        self.b2_1_u3: nn.Module = ResUnit3(
            in_channels=b1_channels, out_channels=b2_channels, bias=bias, padding_mode=padding_mode, inplace=inplace)
        self.b2_2_u2: nn.Module = ResUnit2(
            in_channels=b2_channels, out_channels=b2_channels, bias=bias, padding_mode=padding_mode, inplace=inplace)
        self.b2_3_u2: nn.Module = ResUnit2(
            in_channels=b2_channels, out_channels=b2_channels, bias=bias, padding_mode=padding_mode, inplace=inplace)
        self.b2_4_u4: nn.Module = ResUnit4(in_channels=b2_channels, out_channels=b2_channels,
                                           half=True, bias=bias, padding_mode=padding_mode, inplace=inplace)
        self.b2_5_bn: nn.Module = nn.BatchNorm2d(num_features=b2_channels)
        self.b2_6_re: nn.Module = nn.ReLU(inplace)
        self._out_shapes.append((b2_channels, height // 8, width // 8))

        # Block 3
        b3_channels: int = b2_channels * 2
        self.b3_1_u3: nn.Module = ResUnit3(
            in_channels=b2_channels, out_channels=b3_channels, bias=bias, padding_mode=padding_mode, inplace=inplace)
        self.b3_2_u2: nn.Module = ResUnit2(
            in_channels=b3_channels, out_channels=b3_channels, bias=bias, padding_mode=padding_mode, inplace=inplace)
        self.b3_3_u4: nn.Module = ResUnit4(
            in_channels=b3_channels, out_channels=b3_channels, padding=[0, 1, 2], dilation=[1, 1, 2], bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.b3_4_u4: nn.Module = ResUnit4(
            in_channels=b3_channels, out_channels=b3_channels, padding=[0, 1, 16], dilation=[1, 1, 16], bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.b3_5_u4: nn.Module = ResUnit4(
            in_channels=b3_channels, out_channels=b3_channels, padding=[0, 1, 8], dilation=[1, 1, 8], bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.b3_6_u4: nn.Module = ResUnit4(
            in_channels=b3_channels, out_channels=b3_channels, padding=[0, 1, 4], dilation=[1, 1, 4], bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.b3_7_re: nn.Module = nn.ReLU(inplace)

        # Block 4
        b4_channels: int = b3_channels * 2
        self.b4_1_u5: nn.Module = ResUnit5(
            in_channels=b3_channels, out_channels=b4_channels, bias=bias, padding_mode=padding_mode, inplace=inplace)
        self.b4_2_u4: nn.Module = ResUnit4(
            in_channels=b4_channels, out_channels=b4_channels, padding=[0, 2, 8], dilation=[1, 2, 8], bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.b4_3_u4: nn.Module = ResUnit4(
            in_channels=b4_channels, out_channels=b4_channels, padding=[0, 2, 16], dilation=[1, 2, 16], bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.b4_4_bn: nn.Module = nn.BatchNorm2d(num_features=b4_channels)
        self.b4_5_re: nn.Module = nn.ReLU(inplace)
        self._out_shapes.append((b4_channels, height // 16, width // 16))

    def get_out_shape(self) -> List[Tuple[int, int, int]]:
        """get_out_shape

        Get the shape of the tensor obtained by "forward".

        Returns:
            List[Tuple[int, int, int]]: the shape of the tensor obtained by "forward".
        """
        return self._out_shapes

    def forward(self, x: Tensor) -> List[Tensor]:
        # Block 1
        b1_tx1: Tensor = self.b1_1_bn(x)
        b1_tx2: Tensor = self.b1_2_wl(b1_tx1)
        b1_tx3: Tensor = self.b1_3_mp(b1_tx2)
        b1_tx4: Tensor = self.b1_4_u1(b1_tx3)
        b1_tx5: Tensor = self.b1_5_u2(b1_tx4)
        b1_tx6: Tensor = self.b1_6_u2(b1_tx5)
        b1_tx7: Tensor = self.b1_7_re(b1_tx6)

        # Block 2
        b2_tx1: Tensor = self.b2_1_u3(b1_tx7)
        b2_tx2: Tensor = self.b2_2_u2(b2_tx1)
        b2_tx3: Tensor = self.b2_3_u2(b2_tx2)
        b2_tx4: Tensor = self.b2_4_u4(b2_tx3)
        b2_tx5: Tensor = self.b2_5_bn(b2_tx4)
        b2_tx6: Tensor = self.b2_6_re(b2_tx5)

        # Block 3
        b3_tx1: Tensor = self.b3_1_u3(b2_tx6)
        b3_tx2: Tensor = self.b3_2_u2(b3_tx1)
        b3_tx3: Tensor = self.b3_3_u4(b3_tx2)
        b3_tx4: Tensor = self.b3_4_u4(b3_tx3)
        b3_tx5: Tensor = self.b3_5_u4(b3_tx4)
        b3_tx6: Tensor = self.b3_6_u4(b3_tx5)
        b3_tx7: Tensor = self.b3_7_re(b3_tx6)

        # Block 4
        b4_tx1: Tensor = self.b4_1_u5(b3_tx7)
        b4_tx2: Tensor = self.b4_2_u4(b4_tx1)
        b4_tx3: Tensor = self.b4_3_u4(b4_tx2)
        b4_tx4: Tensor = self.b4_4_bn(b4_tx3)
        b4_tx5: Tensor = self.b4_5_re(b4_tx4)

        return [b1_tx7, b2_tx6, b4_tx5]


class eAspp(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        bias: bool = True,
        padding_mode: str = PADDING_REPLICATE,
        inplace: bool = True
    ) -> None:
        super(eAspp, self).__init__()

        channels, height, width = in_shape

        self.branch_channels: int = BASE_CHANNELS * 4
        self.bias: bool = bias
        self.padding_mode: str = padding_mode
        self.inplace: bool = inplace
        self.channels: int = channels

        self._out_shapes: Tuple[int, int, int] = (
            self.branch_channels, height, width)

        # Branch 1
        self.br1: nn.Module = WeightLayer(
            in_channels=channels, out_channels=self.branch_channels, kernel_size=1, bias=bias, padding_mode=padding_mode, inplace=inplace
        )

        # Branch 2
        self.br2: nn.Module = self.init_branch(3)

        # Branch 3
        self.br3: nn.Module = self.init_branch(6)

        # Branch 4
        self.br4: nn.Module = self.init_branch(12)

        # Branch 5
        self.br5: nn.Module = nn.Sequential(
            nn.AvgPool2d(kernel_size=(height, width)),
            WeightLayer(
                in_channels=channels, out_channels=self.branch_channels, kernel_size=1, bias=bias, padding_mode=padding_mode, inplace=inplace, skip_bn=True
            ),
            nn.Upsample(size=(height, width),
                        mode=UPSAMPLE_BILINEAR, align_corners=True)
        )

        # Final
        self.f_wl: nn.Module = WeightLayer(
            in_channels=self.branch_channels*5, out_channels=self.branch_channels, kernel_size=1,
            bias=bias, padding_mode=padding_mode, inplace=inplace
        )

    def init_branch(self, dilation: int) -> nn.Module:
        return nn.Sequential(
            WeightLayer(
                in_channels=self.channels, out_channels=BASE_CHANNELS, kernel_size=1, bias=self.bias, padding_mode=self.padding_mode, inplace=self.inplace
            ),
            WeightLayer(
                in_channels=BASE_CHANNELS, out_channels=BASE_CHANNELS, kernel_size=3, stride=1, padding=dilation,
                dilation=dilation, bias=self.bias, padding_mode=self.padding_mode, inplace=self.inplace
            ),
            WeightLayer(
                in_channels=BASE_CHANNELS, out_channels=BASE_CHANNELS, kernel_size=3, stride=1, padding=dilation,
                dilation=dilation, bias=self.bias, padding_mode=self.padding_mode, inplace=self.inplace
            ),
            WeightLayer(
                in_channels=BASE_CHANNELS, out_channels=self.branch_channels, kernel_size=1,
                bias=self.bias, padding_mode=self.padding_mode, inplace=self.inplace
            )
        )

    def get_out_shape(self) -> Tuple[int, int, int]:
        """get_out_shape

        Get the shape of the tensor obtained by "forward".

        Returns:
            Tuple[int, int, int]: the shape of the tensor obtained by "forward".
        """
        return self._out_shapes

    def forward(self, x: Tensor) -> Tensor:
        br1_tx1: Tensor = self.br1(x)
        br2_tx1: Tensor = self.br2(x)
        br3_tx1: Tensor = self.br3(x)
        br4_tx1: Tensor = self.br4(x)
        br5_tx1: Tensor = self.br5(x)

        tx2: Tensor = torch.cat(
            (br1_tx1, br2_tx1, br3_tx1, br4_tx1, br5_tx1), dim=1)
        return self.f_wl(tx2)


class AdapnetDecoder(nn.Module):
    def __init__(
        self,
        in_shapes: List[Tuple[int, int, int]],
        out_channels: int,
        use_aux: bool,
        b3_channels: int = None,
        bias: bool = True,
        padding_mode: str = PADDING_REPLICATE,
        inplace: bool = True
    ):
        super(AdapnetDecoder, self).__init__()
        in_channels, in_height, in_width = in_shapes[-1]
        out_height, out_width = in_height * 16, in_width * 16
        d_channels: int = BASE_CHANNELS * 4
        self.use_aux: bool = use_aux

        # Block 1
        self.b1_1_tw: nn.Module = TransposeWeightLayer(
            in_channels, out_channels=d_channels, kernel_size=4, stride=2, padding=1, bias=bias
        )

        # Aux 1
        if self.use_aux is True:
            self.a1_1_wl: nn.Module = WeightLayer(
                in_channels=d_channels, out_channels=out_channels, kernel_size=1, bias=bias, padding_mode=padding_mode, inplace=inplace)
            self.a1_2_us: nn.Module = nn.Upsample(
                size=(out_height, out_width), mode=UPSAMPLE_BILINEAR, align_corners=True)

        # Skip 1
        self.s1_1_sl: nn.Module = AdapnetSkipLayer(
            in_channels=BASE_CHANNELS*8, out_channels=SKIP_CHANNELS, bias=bias, padding_mode=padding_mode)

        # Block 2
        self.b2_1_wl: nn.Module = WeightLayer(
            in_channels=d_channels+SKIP_CHANNELS, out_channels=d_channels, kernel_size=3, padding=1, bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.b2_2_wl: nn.Module = WeightLayer(
            in_channels=d_channels, out_channels=d_channels, kernel_size=3, padding=1, bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.b2_3_tw: nn.Module = TransposeWeightLayer(
            in_channels=d_channels, out_channels=d_channels, kernel_size=4, stride=2, padding=1, bias=bias
        )

        # Aux 2
        if self.use_aux is True:
            self.a2_1_wl: nn.Module = WeightLayer(
                in_channels=d_channels, out_channels=out_channels, kernel_size=1, bias=bias, padding_mode=padding_mode, inplace=inplace)
            self.a2_2_us: nn.Module = nn.Upsample(
                size=(out_height, out_width), mode=UPSAMPLE_BILINEAR, align_corners=True)

        # Skip 2
        self.s2_1_sl: nn.Module = AdapnetSkipLayer(
            in_channels=BASE_CHANNELS*4, out_channels=SKIP_CHANNELS, bias=bias, padding_mode=padding_mode)

        # Block 3
        b3_channels: int = out_channels if b3_channels is None else b3_channels
        self.b3_1_wl: nn.Module = WeightLayer(
            in_channels=d_channels+SKIP_CHANNELS, out_channels=d_channels, kernel_size=3, padding=1, bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.b3_2_wl: nn.Module = WeightLayer(
            in_channels=d_channels, out_channels=d_channels, kernel_size=3, padding=1, bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.b3_3_wl: nn.Module = WeightLayer(
            in_channels=d_channels, out_channels=b3_channels, kernel_size=1, bias=bias, padding_mode=padding_mode, inplace=inplace
        )
        self.b3_4_tw: nn.Module = TransposeWeightLayer(
            in_channels=b3_channels, out_channels=out_channels, kernel_size=8, stride=4, padding=2, bias=bias
        )

    def forward(self, skips: List[Tensor]) -> ADAPNET_DECODER_OUTPUT:
        tx1: Tensor = self.b1_1_tw(skips[-1])

        sk1: Tensor = self.s1_1_sl(skips[1])

        tx2: Tensor = torch.cat((tx1, sk1), dim=1)
        tx3: Tensor = self.b2_1_wl(tx2)
        tx4: Tensor = self.b2_2_wl(tx3)
        tx5: Tensor = self.b2_3_tw(tx4)

        sk2: Tensor = self.s2_1_sl(skips[0])

        tx6: Tensor = torch.cat((tx5, sk2), dim=1)
        tx7: Tensor = self.b3_1_wl(tx6)
        tx8: Tensor = self.b3_2_wl(tx7)
        tx9: Tensor = self.b3_3_wl(tx8)
        tx10: Tensor = self.b3_4_tw(tx9)

        if self.use_aux is True:
            a1x1: Tensor = self.a1_1_wl(tx1)
            a1x2: Tensor = self.a1_2_us(a1x1)

            a2x1: Tensor = self.a2_1_wl(tx5)
            a2x2: Tensor = self.a2_2_us(a2x1)

            return ADAPNET_DECODER_OUTPUT(tx10, a1x2, a2x2)
        else:
            return ADAPNET_DECODER_OUTPUT(tx10, None, None)
