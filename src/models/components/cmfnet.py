import math
from abc import abstractmethod
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.sca_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 4,
                out_channels=dw_channel // 4,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )
        self.sca_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 4,
                out_channels=dw_channel // 4,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x_avg, x_max = x.chunk(2, dim=1)
        x_avg = self.sca_avg(x_avg) * x_avg
        x_max = self.sca_max(x_max) * x_max
        x = torch.cat([x_avg, x_max], dim=1)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class MultiscalePANEncoder(nn.Module):
    def __init__(
        self,
        in_channel: int,
        width: int = 64,
        enc_blk_nums: List = [1, 1],
        middle_blk_num: int = 1,
    ):
        super().__init__()
        self.intro = nn.Conv2d(
            in_channels=in_channel,
            out_channels=width,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.encs = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width

        for num in enc_blk_nums:
            self.encs.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

    def forward(self, pan):
        pan = self.intro(pan)
        encs = []
        for enc, down in zip(self.encs, self.downs):
            pan = enc(pan)
            encs.append(pan)
            pan = down(pan)
        pan = self.middle_blks(pan)
        encs.append(pan)
        return encs


class MultiscaleMSEncoder(nn.Module):
    def __init__(
        self,
        in_channel: int,
        width: int = 64,
        enc_blk_nums: List = [1, 1],
        middle_blk_num: int = 1,
    ):
        super().__init__()
        self.intro = nn.Conv2d(
            in_channels=in_channel,
            out_channels=width * 4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.encs = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width * 4

        for num in enc_blk_nums:
            self.encs.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.ups.append(
                nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2))
            )
            chan = chan // 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

    def forward(self, ms):
        ms = self.intro(ms)
        encs = []
        for enc, up in zip(self.encs, self.ups):
            ms = enc(ms)
            encs.append(ms)
            ms = up(ms)
        ms = self.middle_blks(ms)
        encs.append(ms)
        return encs[::-1]


class UEDM(nn.Module):
    def __init__(
        self,
        img_channel=4,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[1, 1],
        dec_blk_nums=[1, 1],
    ):
        super().__init__()
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=4,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()

        self.decoders = nn.ModuleList()

        self.middle_blks = nn.ModuleList()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2))
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)
        self.ms_encs = MultiscaleMSEncoder(4, width, [1, 1], 1)
        self.pan_encs = MultiscalePANEncoder(1, width, [1, 1], 1)

    def forward(self, ms, pan):
        ms_encs, pan_encs = self.ms_encs(ms), self.pan_encs(pan)
        fuse = 0
        for encoder, down, ms, pan in zip(self.encoders, self.downs, ms_encs[:2], pan_encs[:2]):
            # x = encoder(x)
            # cond = cond_encoder(cond)
            # x = x + cond
            # encs.append(x)
            # x = down(x)
            # cond = cond_down(cond)
            fuse = ms + pan + fuse
            fuse = encoder(fuse)
            fuse = down(fuse)
        fuse = fuse + ms_encs[-1] + pan_encs[-1]
        fuse = self.middle_blks(fuse)
        # print(fuse.shape)

        for decoder, up, ms, pan in zip(
            self.decoders, self.ups, ms_encs[::-1][1:], pan_encs[::-1][1:]
        ):
            fuse = up(fuse)
            fuse = fuse + ms + pan  # uedm6 skip+ms+pan
            fuse = decoder(fuse)

        fuse = self.ending(fuse)
        return fuse


if __name__ == "__main__":
    # x = torch.randn(1, 4, 64, 64)
    # y = torch.randn(1, 1, 256, 256)
    # out = UEDM(width=32, enc_blk_nums=[1, 1], dec_blk_nums=[1, 1])(x, y)
    # assert out.shape == (1, 4, 256, 256)
    x = torch.randn(1, 1, 256, 256)
    n = MultiscalePANEncoder(1, 32)
    for i in n(x):
        print(i.shape)
