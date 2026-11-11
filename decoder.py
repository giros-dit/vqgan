import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, UpSampleBlock, GroupNorm, Swish
from encoder import reduction_plan


class Decoder(nn.Module):
    """
    Decoder del VQGAN: deshace la reducción del encoder (mismo número de bloques,
    con factor (2, 1) o, en 5GmmGesture, (2, 2)). Las ampliaciones van en las últimas etapas.
    """
    def __init__(self, args):
        super(Decoder, self).__init__()
        channels = [512, 256, 256, 128, 128]
        num_res_blocks = 3

        n_up, stride = reduction_plan(args)
        if not 1 <= n_up <= len(channels) - 1:
            raise ValueError(f"Número de ampliaciones no válido: {n_up}")
        scale = (float(stride[0]), float(stride[1]))

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

            # Con x16 y 1D: etapas 1 a 4 (como el original); con x4: etapas 3 y 4
            if i >= len(channels) - n_up:
                layers.append(UpSampleBlock(in_channels, scale=scale))

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, args.image_channels, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

