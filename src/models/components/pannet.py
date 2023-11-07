import kornia
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """Residual block from ResNet."""

    def __init__(self, channels: int) -> None:
        """Initialize.

        Args:
            channels (int): The number of input channels.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Output.
        """
        return x + self.net(x)


class PanNet(nn.Module):
    """[Pixel-level Concatenation] PanNet: A Deep Network Architecture for Pan-Sharpening (https://openaccess.thecvf.com/content_iccv_2017/html/Yang_PanNet_A_Deep_ICCV_2017_paper.html)."""

    def __init__(self, in_channels: int = 5, out_channels: int = 4):
        """Initialize.

        Args:
            in_channels (int, optional): The number of input channels. Defaults to 5.
            out_channels (int, optional): The number of output channels. Defaults to 4.
        """
        super().__init__()
        self.layer_0 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels - 1,
                out_channels=in_channels - 1,
                kernel_size=8,
                stride=4,
                padding=2,
                output_padding=0,
            )
        )
        self.layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.layer_2 = nn.Sequential(
            self.make_layer(ResidualBlock, 4, 32),
            nn.Conv2d(
                in_channels=32,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def make_layer(self, block: nn.Module, num_of_layer: int, channels: int) -> nn.Module:
        """Make a neural network with stacked residual block.

        Args:
            block (nn.Module): The structure of residual block.
            num_of_layer (int): The number of layers.
            channels (int): The dim of features.

        Returns:
            nn.Module: A neural network with stacked residual block.
        """
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x (torch.Tensor): The input tensor of multi-spectral image.
            y (torch.Tensor): The input tensor of panchromatic image.

        Returns:
            torch.Tensor: A tensor of predictions.
        """
        lr_up = F.interpolate(x, y.size()[-2:], mode="bicubic", align_corners=True)
        lr_hp = x - kornia.filters.BoxBlur((5, 5))(x)
        pan_hp = y - kornia.filters.BoxBlur((5, 5))(y)
        lr_u_hp = self.layer_0(lr_hp)
        ms = torch.cat([pan_hp, lr_u_hp], dim=1)
        fea = self.layer_1(ms)
        output = self.layer_2(fea) + lr_up
        return output


if __name__ == "__main__":
    net = PanNet(5, 4)
    x = torch.randn(1, 4, 64, 64)
    y = torch.randn(1, 1, 256, 256)
    out = net(x, y)
    assert out.shape == (1, 4, 256, 256)
