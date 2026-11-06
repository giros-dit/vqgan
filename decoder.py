import math
import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, UpSampleBlock, GroupNorm, Swish


class Decoder(nn.Module):
    """
    Decoder del VQGAN: deshace la reducción del encoder (args.downsample, 16 o 4)
    con UpSampleBlock de factor (2, 1). Las ampliaciones van en las últimas etapas,
    simétricas a las reducciones del encoder.
    """
    def __init__(self, args):
        super(Decoder, self).__init__()
        channels = [512, 256, 256, 128, 128]
        num_res_blocks = 3

        factor = getattr(args, "downsample", 16)
        n_up = int(round(math.log2(factor)))
        if 2 ** n_up != factor or not 1 <= n_up <= len(channels) - 1:
            raise ValueError(f"downsample debe ser 2, 4, 8 o 16, no {factor}")

        in_channels = channels[0]
        layers = [
            nn.Conv2d(args.latent_dim, in_channels, kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels, in_channels),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels)
        ]

        for i in range(len(channels)):
            out_channels = channels[i]
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels

            # Con x16: etapas 1 a 4 (como el original); con x4: etapas 3 y 4
            if i >= len(channels) - n_up:
                layers.append(UpSampleBlock(in_channels))

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, args.image_channels, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

