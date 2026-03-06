import torch
import torch.nn as nn


class Conv2D(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        """Build a Conv-BN-ReLU block."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Apply the block to an input feature map."""
        return self.net(x)


class DSConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        """Build a depthwise-separable Conv-BN-ReLU block."""
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """Apply depthwise then pointwise convolution with normalization."""
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Conv2DOut(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        """Build a plain output convolution without activation."""
        super().__init__()
        self.net = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)

    def forward(self, x):
        """Apply the output convolution."""
        return self.net(x)


class DSConv2DOut(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        """Build a depthwise-separable output convolution."""
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        """Apply depthwise-separable output convolution."""
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MaxPool(nn.Module):
    def __init__(self, k, s):
        """Build a max-pooling layer."""
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=k, stride=s)

    def forward(self, x):
        """Downsample features with max pooling."""
        return self.pool(x)


class DeConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, in_p, out_p=1):
        """Build a transposed-conv upsampling block."""
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=in_p, output_padding=out_p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Apply the upsampling block to input features."""
        return self.net(x)
