import math
import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, DownSampleBlock, GroupNorm, Swish


class Encoder(nn.Module):
    """
    Encoder del VQGAN para grids de canal (C, H, W).
    Reduce el eje vertical (frecuencia / retardo) por args.downsample
    (16 = VQGAN original, 4 = la misma reducción que el VQ-VAE) con DownSampleBlock
    de stride (2, 1). El eje de las columnas (símbolos / estaciones base) no se toca.
    """
    def __init__(self, args):
        super(Encoder, self).__init__()

        channels = [128, 128, 128, 256, 256, 512]
        num_res_blocks = 2

        factor = getattr(args, "downsample", 16)
        n_down = int(round(math.log2(factor)))
        if 2 ** n_down != factor or not 1 <= n_down <= len(channels) - 2:
            raise ValueError(f"downsample debe ser 2, 4, 8 o 16, no {factor}")

        layers = [nn.Conv2d(args.image_channels, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels

            # Las reducciones van en las primeras etapas (con x16: etapas 0 a 3, como el original)
            if i < n_down:
                layers.append(DownSampleBlock(channels[i + 1]))

        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))

        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], args.latent_dim, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)