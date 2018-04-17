import torch
import torch.nn as nn
import numpy as np


class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()

        # Initial convolution layers
        self.conv1 = ConvLayer(1, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128)
        self.conv4 = ConvLayer(128, 256, kernel_size=3, stride=2)
        self.in4 = nn.InstanceNorm2d(256)

        # Residual layers group1
        self.res1 = ResidualBottleneckBlock(256, 288)
        self.res2 = ResidualBottleneckBlock(288, 320)
        self.res3 = ResidualBottleneckBlock(320, 352)
        self.res4 = ResidualBottleneckBlock(352, 384)
        self.res5 = ResidualBottleneckBlock(384, 416)
        self.res6 = ResidualBottleneckBlock(416, 448)
        self.res7 = ResidualBottleneckBlock(448, 480)
        self.res8 = ResidualBottleneckBlock(480, 512)

        # Residual layers group2
        self.res8_d = ResidualBottleneckBlock(512, 480, True)
        self.res7_d = ResidualBottleneckBlock(480, 448, True)
        self.res6_d = ResidualBottleneckBlock(448, 416, True)
        self.res5_d = ResidualBottleneckBlock(416, 384, True)
        self.res4_d = ResidualBottleneckBlock(384, 352, True)
        self.res3_d = ResidualBottleneckBlock(352, 320, True)
        self.res2_d = ResidualBottleneckBlock(320, 288, True)
        self.res1_d = ResidualBottleneckBlock(288, 256, True)

        # Initial deconvolution layers
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in5 = nn.InstanceNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in6 = nn.InstanceNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in7 = nn.InstanceNorm2d(32)
        self.deconv1 = nn.ConvTranspose2d(32, 1, kernel_size=9, stride=1, padding=4)
        self.in8 = nn.InstanceNorm2d(1)

        # Non-linearities

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, X):
        in_X = X
        y = self.relu(self.in1(self.conv1(in_X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.relu(self.in4(self.conv4(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)
        y = self.res7(y)
        y = self.res8(y)
        y = self.res8_d(y)
        y = self.res7_d(y)
        y = self.res6_d(y)
        y = self.res5_d(y)
        y = self.res4_d(y)
        y = self.res3_d(y)
        y = self.res2_d(y)
        y = self.res1_d(y)
        y = self.relu(self.in5(self.deconv4(y)))
        y = self.relu(self.in6(self.deconv3(y)))
        y = self.relu(self.in7(self.deconv2(y)))
        y = self.in8(self.deconv1(y))
        y = self.tanh(y)
        return y

class Discriminator(nn.Module):
    def __init__(self, DIM):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(1, DIM, 3, 2, padding=1)
        self.conv2 = nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1)
        self.conv4 = nn.Conv2d(4 * DIM, 8 * DIM, 3, 2, padding=1)
        self.conv5 = nn.Conv2d(8 * DIM, 16 * DIM, 3, 2, padding=1)

        self.elu = nn.ELU()
        self.linear = nn.Linear(8 * 8 * 16 * DIM, 1)
        self.DIM = DIM

    def forward(self, input):
        y = self.elu(self.conv1(input))
        y = self.elu(self.conv2(y))
        y = self.elu(self.conv3(y))
        y = self.elu(self.conv4(y))
        y = self.elu(self.conv5(y))
        y = y.view(y.size(0), -1)
        output = self.linear(y)
        return output


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = InstanceNormalization(channels)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = InstanceNormalization(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class ResidualBottleneckBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, Deconv=False):

        super(ResidualBottleneckBlock, self).__init__()

        self.conv_shortcuts = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=False)
        self.instance_shortcuts = nn.InstanceNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, np.int(out_channels / 4), kernel_size=1, stride=1, bias=False)
        self.in1 = nn.InstanceNorm2d(np.int(out_channels / 4))
        self.conv2 = nn.Conv2d(np.int(out_channels / 4), np.int(out_channels / 4), kernel_size=3, stride=1, bias=False)
        self.in2 = nn.InstanceNorm2d(np.int(out_channels / 4))
        self.conv3 = nn.Conv2d(np.int(out_channels / 4), out_channels, kernel_size=1, stride=1, bias=False)
        self.in3 = nn.InstanceNorm2d(out_channels)

        self.deconv_shortcuts = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, bias=False)
        self.instance_d_shortcuts = nn.InstanceNorm2d(out_channels)
        self.deconv1 = nn.ConvTranspose2d(in_channels, np.int(out_channels / 4), kernel_size=1, stride=1, bias=False)
        self.in1_d = nn.InstanceNorm2d(np.int(out_channels / 4))
        self.deconv2 = nn.ConvTranspose2d(np.int(out_channels / 4), np.int(out_channels / 4), kernel_size=3, stride=1, bias=False)
        self.in2_d = nn.InstanceNorm2d(np.int(out_channels / 4))
        self.deconv3 = nn.ConvTranspose2d(np.int(out_channels / 4), out_channels, kernel_size=1, stride=1, bias=False)
        self.in3_d = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.Deconv = Deconv

    def forward(self, x):
        if self.Deconv:
            residual = self.instance_d_shortcuts(self.deconv_shortcuts(x))
            out = self.relu(self.in1_d(self.deconv1(x)))
            out = self.relu(self.in2_d(self.deconv2(out)))
            out = self.in3_d(self.deconv3(out))
            out = out + residual
        else:
            residual = self.instance_shortcuts(self.conv_shortcuts(x))
            out = self.relu(self.in1(self.conv1(x)))
            out = self.relu(self.in2(self.conv2(out)))
            out = self.in3(self.conv3(out))
            out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class InstanceNormalization(torch.nn.Module):
    """InstanceNormalization
    Improves convergence of neural-style.
    ref: https://arxiv.org/pdf/1607.08022.pdf
    """

    def __init__(self, dim, eps=1e-9):
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out
