import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode="bicubic", align_corners=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = x + self.conv2(self.relu(self.conv1(x)))
        return x


class BasicUnit(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3):
        super(BasicUnit, self).__init__()
        p = kernel_size // 2
        self.basic_unit = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=p, bias=False),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, out_channels, kernel_size, padding=p, bias=False),
        )

    def forward(self, input):
        return self.basic_unit(input)


# LRBlock is called MSBlock in our paper


class LRBlock(nn.Module):
    def __init__(self, ms_channels, n_feat):
        super(LRBlock, self).__init__()
        self.get_LR = BasicUnit(ms_channels, n_feat, ms_channels)
        self.get_HR_residual = BasicUnit(ms_channels, n_feat, ms_channels)
        self.prox = BasicUnit(ms_channels, n_feat, ms_channels)

    def forward(self, HR, LR):
        _, _, M, N = HR.shape
        _, _, m, n = LR.shape

        LR_hat = upsample(self.get_LR(HR), m, n)
        LR_Residual = LR - LR_hat
        HR_Residual = upsample(self.get_HR_residual(LR_Residual), M, N)
        HR = self.prox(HR + HR_Residual)
        return HR


class PANBlock(nn.Module):
    def __init__(self, ms_channels, pan_channels, n_feat, kernel_size):
        super(PANBlock, self).__init__()
        self.get_PAN = BasicUnit(ms_channels, n_feat, pan_channels, kernel_size)
        self.get_HR_residual = BasicUnit(pan_channels, n_feat, ms_channels, kernel_size)
        self.prox = BasicUnit(ms_channels, n_feat, ms_channels, kernel_size)

    def forward(self, HR, PAN):
        PAN_hat = self.get_PAN(HR)
        PAN_Residual = PAN - PAN_hat
        HR_Residual = self.get_HR_residual(PAN_Residual)
        HR = self.prox(HR + HR_Residual)
        return HR


class GPPNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 4,
    ):
        super(GPPNN, self).__init__()
        n_feat = 64
        n_layer = 8
        self.lr_blocks = nn.ModuleList(
            [LRBlock(in_channels - 1, n_feat) for i in range(n_layer)]
        )
        self.pan_blocks = nn.ModuleList(
            [PANBlock(in_channels - 1, 1, n_feat, 1) for i in range(n_layer)]
        )

    def forward(self, ms, pan):
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape
        HR = upsample(ms, M, N)

        for i in range(len(self.lr_blocks)):
            HR = self.lr_blocks[i](HR, ms)
            HR = self.pan_blocks[i](HR, pan)
        return HR


if __name__ == "__main__":
    net = GPPNN(5, 4)
    x = torch.randn(1, 4, 64, 64)
    y = torch.randn(1, 1, 256, 256)
    out = net(x, y)
    assert out.shape == (1, 4, 256, 256)
