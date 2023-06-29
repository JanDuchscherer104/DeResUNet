from copy import deepcopy

import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from torchvision.models import resnet50


class ResNet4Channel(nn.Module):
    def __init__(self):
        super(ResNet4Channel, self).__init__()
        resnet = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        weights = resnet.conv1.weight.clone()

        # RGB channels
        self.conv1_rgb = resnet.conv1
        self.bn1_rgb = resnet.bn1
        self.relu_rgb = resnet.relu
        self.maxpool_rgb = resnet.maxpool
        self.bn1_rgb = resnet.bn1
        self.relu_rgb = resnet.relu
        self.maxpool_rgb = resnet.maxpool
        self.layer1_rgb = resnet.layer1
        self.layer2_rgb = resnet.layer2
        self.layer3_rgb = resnet.layer3
        self.layer4_rgb = resnet.layer4

        # Depth channel
        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_d.weight = nn.Parameter(
            torch.mean(weights[:, :2, :, :], dim=1, keepdim=True)
        )
        self.bn1_d = nn.BatchNorm2d(64)
        self.relu_d = nn.ReLU(inplace=True)
        self.maxpool_d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_d = deepcopy(resnet.layer1)
        self.layer2_d = deepcopy(resnet.layer2)
        self.layer3_d = deepcopy(resnet.layer3)
        self.layer4_d = deepcopy(resnet.layer4)

    def forward(self, x):
        x_rgb, x_d = x[:, :3, :, :], x[:, 3:, :, :]

        # RGB channels
        x_rgb = self.conv1_rgb(x_rgb)
        x_rgb = self.bn1_rgb(x_rgb)
        x0_rgb = self.relu_rgb(x_rgb)
        x_rgb = self.maxpool_rgb(x0_rgb)
        x1_rgb = self.layer1_rgb(x_rgb)
        x2_rgb = self.layer2_rgb(x1_rgb)
        x3_rgb = self.layer3_rgb(x2_rgb)
        x4_rgb = self.layer4_rgb(x3_rgb)

        # Depth channel
        x_d = self.conv1_d(x_d)
        x_d = self.bn1_d(x_d)
        x0_d = self.relu_d(x_d)
        x_d = self.maxpool_d(x0_d)
        x1_d = self.layer1_d(x_d)
        x2_d = self.layer2_d(x1_d)
        x3_d = self.layer3_d(x2_d)
        x4_d = self.layer4_d(x3_d)

        # Concatenate the outputs
        x0 = torch.cat((x0_rgb, x0_d), dim=1)
        x1 = torch.cat((x1_rgb, x1_d), dim=1)
        x2 = torch.cat((x2_rgb, x2_d), dim=1)
        x3 = torch.cat((x3_rgb, x3_d), dim=1)
        x4 = torch.cat((x4_rgb, x4_d), dim=1)

        return x0, x1, x2, x3, x4


class ConvNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, in_channels_up=None, out_channels_up=None
    ):
        super().__init__()
        if in_channels_up is None:
            in_channels_up = in_channels
        if out_channels_up is None:
            out_channels_up = out_channels
        self.upconv = nn.ConvTranspose2d(
            in_channels_up, out_channels_up, kernel_size=2, stride=2
        )
        self.conv1 = ConvNormRelu(in_channels, out_channels)
        self.conv2 = ConvNormRelu(out_channels, out_channels)

    def forward(self, x, x_cat):
        x = self.upconv(x)
        x = torch.cat([x, x_cat], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpConvPxShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, in_channels_up=None):
        super().__init__()
        if in_channels_up is not None:
            in_channels = in_channels + in_channels_up

        self.upscale = nn.PixelShuffle(2)  # upscale factor is 2 => # channels / 4
        self.conv1 = ConvNormRelu(in_channels, out_channels)
        self.conv2 = ConvNormRelu(out_channels, out_channels)

    def forward(self, x, x_cat):
        x = self.upscale(x)
        x = torch.cat([x, x_cat], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResUNet(nn.Module):
    def __init__(self, num_classes):
        super(ResUNet, self).__init__()

        self.num_classes = num_classes
        self.resnet = ResNet4Channel()
        self.bottleneck = self._make_bottle_neck(4096, 2048)
        self.upconv3 = UpConvPxShuffle(2048, 1024, in_channels_up=512)
        self.upconv2 = UpConvPxShuffle(1024, 512, in_channels_up=256)
        self.upconv1 = UpConvPxShuffle(512, 256, in_channels_up=128)
        self.upconv0 = UpConvPxShuffle(128, 64, in_channels_up=64)
        self.last_upconv = nn.Sequential(
            nn.PixelShuffle(2),
            ConvNormRelu(16, 8),
        )
        self.last_conv = nn.Conv2d(8, self.num_classes, kernel_size=1, stride=1)

    def _make_bottle_neck(self, in_channels, out_channels):
        return nn.Sequential(
            ConvNormRelu(in_channels, out_channels),
            ConvNormRelu(out_channels, out_channels),
        )

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.resnet(x)
        x = self.bottleneck(x4)
        x = self.upconv3(x, x3)
        x = self.upconv2(x, x2)
        x = self.upconv1(x, x1)
        x = self.upconv0(x, x0)
        x = self.last_upconv(x)
        return self.last_conv(x)

    def summary(self, input_size):
        summary(self, input_size)


if __name__ == "__main__":
    model = ResUNet(50)
    x = model(torch.randn(1, 4, 480, 640))
    summary(model, (4, 480, 640))
    print(model)
    print(x.shape)
