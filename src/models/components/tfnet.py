import torch.nn as nn
import torch
import torch.nn.functional as F


class TFNet(nn.Module):
    def __init__(self, in_channels: int = 5, out_channels: int = 4):
        super(TFNet, self).__init__()
        self.encoder1_pan = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.PReLU(),
        )
        self.encoder2_pan = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2),
            nn.PReLU(),
        )
        self.encoder1_lr = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels - 1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.PReLU(),
        )
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2),
            nn.PReLU(),
        )
        self.fusion1 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.PReLU(),
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2),
            nn.PReLU(),
        )
        self.restore1 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PReLU(),
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=2, stride=2
            ),
            nn.PReLU(),
        )
        self.restore2 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.PReLU(),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=2, stride=2
            ),
            nn.PReLU(),
        )
        self.restore3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=in_channels - 1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, x_lr, x_pan):
        # torch.nn.functional.interpolate(x_lr, scale_factor=cfg.scale, mode='bicubic')
        x_lr = F.interpolate(
            x_lr, x_pan.size()[-2:], mode="bicubic", align_corners=True
        )
        encoder1_pan = self.encoder1_pan(x_pan)
        encoder1_lr = self.encoder1_lr(x_lr)

        encoder2_pan = self.encoder2_pan(encoder1_pan)
        encoder2_lr = self.encoder2_lr(encoder1_lr)

        fusion1 = self.fusion1(torch.cat((encoder2_pan, encoder2_lr), dim=1))
        fusion2 = self.fusion2(fusion1)

        restore1 = self.restore1(fusion2)
        restore2 = self.restore2(torch.cat((restore1, fusion1), dim=1))
        restore3 = self.restore3(
            torch.cat((restore2, encoder1_lr, encoder1_pan), dim=1)
        )

        return restore3


if __name__ == "__main__":
    net = TFNet(5, 4)
    x = torch.randn(1, 4, 64, 64)
    y = torch.randn(1, 1, 256, 256)
    out = net(x, y)
    assert out.shape == (1, 4, 256, 256)
