"""
PatchGAN (Isola et al., 2017) adaptado a grids de canal (C, H, W).
Kernel (4, 1): mira 4 filas seguidas (frecuencia / retardo) de una sola columna.
El eje de las columnas (símbolos / estaciones base) no se reduce nunca.

norm = "batch": BatchNorm tras cada convolución intermedia (el original)
norm = "none":  sin normalización (las convoluciones llevan bias)
"""

import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, args, num_filters_last=64, n_layers=3, norm=None):
        super(Discriminator, self).__init__()
        if norm is None:
            norm = getattr(args, "disc_norm", "batch")
        if norm not in ("batch", "none"):
            raise ValueError(f"norm debe ser 'batch' o 'none', no '{norm}'")
        use_bn = norm == "batch"
        nf = num_filters_last

        layers = [
            nn.Conv2d(args.image_channels, nf, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        mult = 1
        for i in range(1, n_layers + 1):
            mult_last = mult
            mult = min(2 ** i, 8)
            stride_h = 2 if i < n_layers else 1
            layers.append(nn.Conv2d(nf * mult_last, nf * mult, kernel_size=(4, 1),
                                    stride=(stride_h, 1), padding=(1, 0), bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(nf * mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(nf * mult, 1, kernel_size=(4, 1), stride=1, padding=(1, 0)))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)