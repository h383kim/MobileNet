import torch
from torch import nn

class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, bias=False, **kwargs),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6()
        )

    def forward(self, x):
        return self.conv(x)


class Depthwise_Separable(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.depthwise_conv = BasicConv(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch)
        self.pointwise_conv = BasicConv(in_ch, out_ch, kernel_size=1, stride=1) # stride is 1 and padding is 0 in pointwise conv, always

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class MobileNetV1(nn.Module):
    def __init__(self, width_multiplier, num_classes=10):
        super().__init__()

        self.alpha = width_multiplier

        self.stem = BasicConv(in_ch=3, out_ch=int(32*self.alpha), kernel_size=3, stride=2, padding=1)

        self.conv_block = nn.Sequential(
            Depthwise_Separable(int(32*self.alpha), int(64*self.alpha)),
            Depthwise_Separable(int(64*self.alpha), int(128*self.alpha), stride=2),
            Depthwise_Separable(int(128*self.alpha), int(128*self.alpha)),
            Depthwise_Separable(int(128*self.alpha), int(256*self.alpha), stride=2),
            Depthwise_Separable(int(256*self.alpha), int(256*self.alpha)),
            Depthwise_Separable(int(256*self.alpha), int(512*self.alpha), stride=2),
            Depthwise_Separable(int(512*self.alpha), int(512*self.alpha)),
            Depthwise_Separable(int(512*self.alpha), int(512*self.alpha)),
            Depthwise_Separable(int(512*self.alpha), int(512*self.alpha)),
            Depthwise_Separable(int(512*self.alpha), int(512*self.alpha)),
            Depthwise_Separable(int(512*self.alpha), int(512*self.alpha)),
            Depthwise_Separable(int(512*self.alpha), int(1024*self.alpha), stride=2),
            Depthwise_Separable(int(1024*self.alpha), int(1024*self.alpha)),
            nn.AvgPool2d(7)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(1024*self.alpha), num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.conv_block(x)
        x = self.classifier(x)
        return x
            