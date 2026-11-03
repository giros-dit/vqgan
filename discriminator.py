"""
PatchGAN Discriminator adaptado para matrices CSI de GIROS (2, 544, 14).
Utiliza kernel asimétrico (4, 1) con padding (1, 0):
- Evalúa parches a lo largo del eje de subportadoras (frecuencia).
- Mantiene invariable la dimensión temporal de 14 símbolos OFDM (stride=1, kernel=1).
"""

import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, args, num_filters_last=64, n_layers=3):
        super(Discriminator, self).__init__()

        
        layers = [
            nn.Conv2d(
                args.image_channels,
                num_filters_last,
                kernel_size=(4, 1),
                stride=(2, 1),
                padding=(1, 0)
            ),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        num_filters_mult = 1

        
        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            stride_h = 2 if i < n_layers else 1

            layers += [
                nn.Conv2d(
                    num_filters_last * num_filters_mult_last,
                    num_filters_last * num_filters_mult,
                    kernel_size=(4, 1),
                    stride=(stride_h, 1),
                    padding=(1, 0),
                    bias=False
                ),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        
        layers.append(
            nn.Conv2d(
                num_filters_last * num_filters_mult,
                1,
                kernel_size=(4, 1),
                stride=1,
                padding=(1, 0)
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
