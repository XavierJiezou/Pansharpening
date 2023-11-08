import torch.nn as nn
import torch
import torch.nn.functional as F


class MSDCNN(nn.Module):
    def __init__(self, in_channels: int = 5, out_channels: int = 4):
        super(MSDCNN, self).__init__()
        self.shallow_conv_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=9, stride=1, padding=4
        )
        self.shallow_conv_2 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0
        )
        self.shallow_conv_3 = nn.Conv2d(
            in_channels=32,
            out_channels=in_channels - 1,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.relu = nn.ReLU()
        self.deep_conv_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=60, kernel_size=7, stride=1, padding=3
        )
        self.deep_conv_1_sacle_1 = nn.Conv2d(
            in_channels=60, out_channels=20, kernel_size=3, stride=1, padding=1
        )
        self.deep_conv_1_sacle_2 = nn.Conv2d(
            in_channels=60, out_channels=20, kernel_size=5, stride=1, padding=2
        )
        self.deep_conv_1_sacle_3 = nn.Conv2d(
            in_channels=60, out_channels=20, kernel_size=7, stride=1, padding=3
        )
        self.deep_conv_2 = nn.Conv2d(
            in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1
        )
        self.deep_conv_2_sacle_1 = nn.Conv2d(
            in_channels=30, out_channels=10, kernel_size=3, stride=1, padding=1
        )
        self.deep_conv_2_sacle_2 = nn.Conv2d(
            in_channels=30, out_channels=10, kernel_size=5, stride=1, padding=2
        )
        self.deep_conv_2_sacle_3 = nn.Conv2d(
            in_channels=30, out_channels=10, kernel_size=7, stride=1, padding=3
        )
        self.deep_conv_3 = nn.Conv2d(
            in_channels=30,
            out_channels=in_channels - 1,
            kernel_size=5,
            stride=1,
            padding=2,
        )

    def forward(self, x, y):
        x = F.interpolate(x, y.size()[-2:], mode="bicubic", align_corners=True)
        in_put = torch.cat([x, y], -3)

        shallow_fea = self.relu(self.shallow_conv_1(in_put))
        shallow_fea = self.relu(self.shallow_conv_2(shallow_fea))
        shallow_out = self.shallow_conv_3(shallow_fea)

        deep_fea = self.relu(self.deep_conv_1(in_put))
        deep_fea_scale1 = self.relu(self.deep_conv_1_sacle_1(deep_fea))
        deep_fea_scale2 = self.relu(self.deep_conv_1_sacle_2(deep_fea))
        deep_fea_scale3 = self.relu(self.deep_conv_1_sacle_3(deep_fea))
        deep_fea_scale = torch.cat(
            [deep_fea_scale1, deep_fea_scale2, deep_fea_scale3], -3
        )
        deep_fea_1 = torch.add(deep_fea, deep_fea_scale)
        deep_fea_2 = self.relu(self.deep_conv_2(deep_fea_1))
        deep_fea_2_scale1 = self.relu(self.deep_conv_2_sacle_1(deep_fea_2))
        deep_fea_2_scale2 = self.relu(self.deep_conv_2_sacle_2(deep_fea_2))
        deep_fea_2_scale3 = self.relu(self.deep_conv_2_sacle_3(deep_fea_2))
        deep_fea_2_scale = torch.cat(
            [deep_fea_2_scale1, deep_fea_2_scale2, deep_fea_2_scale3], -3
        )
        deep_fea_3 = torch.add(deep_fea_2, deep_fea_2_scale)
        deep_out = self.deep_conv_3(deep_fea_3)

        out = deep_out + shallow_out

        return out


if __name__ == "__main__":
    net = MSDCNN(5, 4)
    x = torch.randn(1, 4, 64, 64)
    y = torch.randn(1, 1, 256, 256)
    out = net(x, y)
    assert out.shape == (1, 4, 256, 256)
