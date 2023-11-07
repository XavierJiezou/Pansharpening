import torch
from torch import nn
from torch.nn import functional as F


class PNN(nn.Module):
    """[Pixel-level Concatenation] Pansharpening by Convolutional Neural Networks (https://www.mdpi.com/2072-4292/8/7/594)."""

    def __init__(self, in_channels: int = 5, out_channels: int = 4) -> None:
        """Initialize.

        Args:
            in_channels (int, optional): The number of input channels. Defaults to 5.
            out_channels (int, optional): The number of output channels. Defaults to 4.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 9, 1, 4),
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 5, 1, 2),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x (torch.Tensor): The input tensor of multi-spectral image.
            y (torch.Tensor): The input tensor of panchromatic image.

        Returns:
            torch.Tensor: A tensor of predictions.
        """
        x = F.interpolate(x, y.size()[-2:], mode="bicubic", align_corners=True)
        z = torch.cat([x, y], -3)
        return self.net(z)


if __name__ == "__main__":
    net = PNN(5, 4)
    x = torch.randn(1, 4, 64, 64)
    y = torch.randn(1, 1, 256, 256)
    out = net(x, y)
    assert out.shape == (1, 4, 256, 256)
