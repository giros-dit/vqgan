import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)


class UpSampleBlock(nn.Module):
    """
    Bloque de aumento. scale=(2, 1): duplica solo el eje vertical (lo de siempre).
    scale=(2, 2): duplica los dos ejes (5GmmGesture).
    """
    def __init__(self, channels, scale=(2.0, 1.0)):
        super(UpSampleBlock, self).__init__()
        self.scale = tuple(float(s) for s in scale)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        return self.conv(x)


class DownSampleBlock(nn.Module):
    """
    Bloque de reducción. stride=(2, 1): reduce a la mitad solo el eje vertical (lo de siempre).
    stride=(2, 2): reduce a la mitad los dos ejes (5GmmGesture).
    """
    def __init__(self, channels, stride=(2, 1)):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        return self.conv(x)


class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels

        self.gn = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c) ** (-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        return x + self.proj_out(A)












