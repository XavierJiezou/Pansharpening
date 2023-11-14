import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionNet(nn.Module):
    def __init__(self, num_spectral=4, num_res=4, num_fm=32):
        super(FusionNet, self).__init__()
        self.num_spectral = num_spectral
        self.num_res = num_res
        self.num_fm = num_fm

        self.pan_concat = nn.Sequential(
            nn.Conv2d(self.num_spectral, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.residual_blocks = self.make_res_blocks()

        self.output_layer = nn.Conv2d(
            num_fm, num_spectral, kernel_size=3, stride=1, padding=1
        )

    def make_res_blocks(self):
        layers = []
        for _ in range(self.num_res):
            layers.extend(
                [
                    nn.Conv2d(
                        self.num_fm, self.num_fm, kernel_size=3, stride=1, padding=1
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        self.num_fm, self.num_fm, kernel_size=3, stride=1, padding=1
                    ),
                ]
            )
        return nn.Sequential(*layers)

    def forward(self, lms, pan):
        lms = F.interpolate(lms, pan.size()[-2:], mode="bicubic", align_corners=True)
        pan_concat = torch.cat([pan] * self.num_spectral, dim=1)
        ms = pan_concat - lms

        rs = self.pan_concat(ms)
        rs_out = rs

        for block in self.residual_blocks:
            rs = block(rs)
            rs_out = rs_out + rs

        rs = self.output_layer(rs_out)

        return rs


if __name__ == "__main__":
    net = FusionNet(4, 4, 32)
    x = torch.randn(1, 4, 64, 64)
    y = torch.randn(1, 1, 256, 256)
    out = net(x, y)
    assert out.shape == (1, 4, 256, 256)
