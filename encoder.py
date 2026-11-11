import math
import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, DownSampleBlock, GroupNorm, Swish


def reduction_plan(args):
    """
    Cuántos bloques de reducción y con qué stride.
      - Normal: log2(downsample) bloques de stride (2, 1) -> solo el eje vertical.
      - down2d (5GmmGesture): log2(downsample) / 2 bloques de stride (2, 2) -> los dos ejes,
        con el mismo factor total (x16 = 4 x 4, x4 = 2 x 2).
    """
    factor = getattr(args, "downsample", 16)
    down2d = getattr(args, "down2d", False)
    n_total = int(round(math.log2(factor)))
    if 2 ** n_total != factor:
        raise ValueError(f"downsample debe ser una potencia de 2, no {factor}")
    if down2d:
        if n_total % 2 != 0:
            raise ValueError(f"Con reducción en los dos ejes, downsample debe ser 4 o 16, no {factor}")
        return n_total // 2, (2, 2)
    return n_total, (2, 1)


class Encoder(nn.Module):
    """
    Encoder del VQGAN para grids de canal (C, H, W).
    Reduce el eje vertical por args.downsample (16 = VQGAN original, 4 = como el VQ-VAE).
    Si args.down2d (5GmmGesture), reparte la reducción entre los dos ejes.
    """
    def __init__(self, args):
        super(Encoder, self).__init__()

        channels = [128, 128, 128, 256, 256, 512]
        num_res_blocks = 2

        n_down, stride = reduction_plan(args)
        if not 1 <= n_down <= len(channels) - 2:
            raise ValueError(f"Número de reducciones no válido: {n_down}")

        layers = [nn.Conv2d(args.image_channels, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels

            # Las reducciones van en las primeras etapas (con x16 y 1D: etapas 0 a 3, como el original)
            if i < n_down:
                layers.append(DownSampleBlock(channels[i + 1], stride=stride))

        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))

        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], args.latent_dim, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)