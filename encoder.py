import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, DownSampleBlock, GroupNorm, Swish


class Encoder(nn.Module):
    """
    Encoder adaptado para CSI de GIROS (2, 544, 14):
    - Entrada: (B, image_channels=2, 544, 14)
    - 4 etapas de DownSampleBlock asimétrico (stride=(2, 1))
    - Salida latente: (B, latent_dim, 34, 14) -> Factor 16 en frecuencia, 0 en tiempo
    """
    def __init__(self, args):
        super(Encoder, self).__init__()
        
        channels = [128, 128, 128, 256, 256, 512]
        num_res_blocks = 2

        layers = [nn.Conv2d(args.image_channels, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels


            if i != len(channels) - 2:
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