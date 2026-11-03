import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, UpSampleBlock, GroupNorm, Swish


class Decoder(nn.Module):
    """
    Decoder adaptado para matrices CSI de GIROS:
    - Entrada latente: (B, latent_dim, 34, 14)
    - 4 etapas de UpSampleBlock asimétrico (scale_factor=(2.0, 1.0))
    - Salida reconstruida: (B, image_channels=2, 544, 14)
    """
    def __init__(self, args):
        super(Decoder, self).__init__()
        channels = [512, 256, 256, 128, 128]
        num_res_blocks = 3

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
            

            if i != 0:
                layers.append(UpSampleBlock(in_channels))

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, args.image_channels, kernel_size=3, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

