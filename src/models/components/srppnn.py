import os
import torch.nn as nn
import torch
import kornia
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch
import warnings

warnings.filterwarnings("ignore")


class Residual_Block(nn.Module):
    def __init__(self, channels):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        fea = self.relu(self.conv1(x))
        fea = self.conv2(fea)
        result = fea + x
        return result


class SRPPNN(nn.Module):
    def __init__(self, in_channels: int = 5, out_channels: int = 4):
        super(SRPPNN, self).__init__()
        self.pan_extract_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            self.make_layer(Residual_Block, 10, 64),
            nn.Conv2d(
                in_channels=64,
                out_channels=in_channels - 1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.pan_extract_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels - 1,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            self.make_layer(Residual_Block, 10, 64),
            nn.Conv2d(
                in_channels=64,
                out_channels=in_channels - 1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=2),
        )
        self.ms_up_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels - 1,
                out_channels=in_channels - 1 * 4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PixelShuffle(2),
        )
        self.ms_up_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels - 1,
                out_channels=in_channels - 1 * 4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PixelShuffle(2),
        )

        self.conv_mul_pre_p1 = nn.Conv2d(
            in_channels=in_channels - 1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.img_mul_p1_layer = self.make_layer(Residual_Block, 4, 32)
        self.conv_mul_post_p1 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.mul_grad_p1 = nn.Conv2d(
            in_channels=32,
            out_channels=in_channels - 1,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.mul_grad_p2 = nn.Conv2d(
            in_channels=32,
            out_channels=in_channels - 1,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_pre_p1 = nn.Conv2d(
            in_channels=in_channels - 1 + 1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_pre_p2 = nn.Conv2d(
            in_channels=in_channels - 1 + 1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.img_p1_layer = self.make_layer(Residual_Block, 4, 32)
        self.conv_post_p1 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv_post_p2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.grad_p1 = nn.Conv2d(
            in_channels=32,
            out_channels=in_channels - 1,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.grad_p2 = nn.Conv2d(
            in_channels=32,
            out_channels=in_channels - 1,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_mul_pre_p2 = nn.Conv2d(
            in_channels=in_channels - 1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.img_mul_p2_layer = self.make_layer(Residual_Block, 4, 32)
        self.conv_mul_post_p2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.img_p2_layer = self.make_layer(Residual_Block, 4, 32)

    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        inputs_mul_up_p1 = F.interpolate(
            x, scale_factor=2, mode="bicubic", align_corners=True
        )
        inputs_mul_up_p2 = F.interpolate(
            x, scale_factor=4, mode="bicubic", align_corners=True
        )
        inputs_pan = y
        inputs_pan_blur = kornia.filters.GaussianBlur2d((11, 11), (1, 1))(y)
        inputs_pan_down_p1 = F.interpolate(
            inputs_pan_blur, scale_factor=0.5, mode="bicubic", align_corners=True
        )
        pre_inputs_mul_p1_feature = self.conv_mul_pre_p1(x)
        x = pre_inputs_mul_p1_feature
        x = self.img_mul_p1_layer(x)
        post_inputs_mul_p1_feature = self.conv_mul_post_p1(x)
        inputs_mul_p1_feature = pre_inputs_mul_p1_feature + post_inputs_mul_p1_feature
        inputs_mul_p1_feature_bic = F.interpolate(
            inputs_mul_p1_feature, scale_factor=2, mode="bicubic", align_corners=True
        )
        net_img_p1_sr = self.mul_grad_p1(inputs_mul_p1_feature_bic) + inputs_mul_up_p1
        inputs_p1 = torch.cat([net_img_p1_sr, inputs_pan_down_p1], -3)

        pre_inputs_p1_feature = self.conv_pre_p1(inputs_p1)
        x = pre_inputs_p1_feature
        x = self.img_p1_layer(x)
        post_inputs_p1_feature = self.conv_post_p1(x)
        inputs_p1_feature = pre_inputs_p1_feature + post_inputs_p1_feature

        inputs_pan_down_p1_blur = kornia.filters.GaussianBlur2d((11, 11), (1, 1))(
            inputs_pan_down_p1
        )
        inputs_pan_hp_p1 = inputs_pan_down_p1 - inputs_pan_down_p1_blur
        net_img_p1 = (
            self.grad_p1(inputs_p1_feature) + inputs_mul_up_p1 + inputs_pan_hp_p1
        )

        pre_inputs_mul_p2_feature = self.conv_mul_pre_p2(net_img_p1)
        x = pre_inputs_mul_p2_feature
        x = self.img_mul_p2_layer(x)
        post_inputs_mul_p2_feature = self.conv_mul_post_p2(x)
        inputs_mul_p2_feature = pre_inputs_mul_p2_feature + post_inputs_mul_p2_feature
        inputs_mul_p2_feature_bic = F.interpolate(
            inputs_mul_p2_feature, scale_factor=2, mode="bicubic", align_corners=True
        )
        net_img_p2_sr = self.mul_grad_p2(inputs_mul_p2_feature_bic) + inputs_mul_up_p2
        inputs_p2 = torch.cat([net_img_p2_sr, inputs_pan], -3)

        pre_inputs_p2_feature = self.conv_pre_p2(inputs_p2)
        x = pre_inputs_p2_feature
        x = self.img_p2_layer(x)
        post_inputs_p2_feature = self.conv_post_p2(x)
        inputs_p2_feature = pre_inputs_p2_feature + post_inputs_p2_feature

        inputs_pan_hp_p2 = inputs_pan - inputs_pan_blur
        net_img_p2 = (
            self.grad_p2(inputs_p2_feature) + inputs_mul_up_p2 + inputs_pan_hp_p2
        )
        return net_img_p2


if __name__ == "__main__":
    net = SRPPNN(5, 4)
    x = torch.randn(1, 4, 64, 64)
    y = torch.randn(1, 1, 256, 256)
    out = net(x, y)
    assert out.shape == (1, 4, 256, 256)
