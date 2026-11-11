"""
PatchGAN (Isola et al., 2017) adaptado a grids de canal (C, H, W).
- Normal: kernel (4, 1) y stride (2, 1): mira 4 filas seguidas de una sola columna.
- down2d (5GmmGesture): kernel (4, 4) y stride (2, 2), el PatchGAN 2D original.

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

        if getattr(args, "down2d", False):
            kernel, stride2, pad = (4, 4), (2, 2), (1, 1)
        else:
            kernel, stride2, pad = (4, 1), (2, 1), (1, 0)

        layers = [
            nn.Conv2d(args.image_channels, nf, kernel_size=kernel, stride=stride2, padding=pad),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        mult = 1
        for i in range(1, n_layers + 1):
            mult_last = mult
            mult = min(2 ** i, 8)
            stride = stride2 if i < n_layers else (1, 1)
            layers.append(nn.Conv2d(nf * mult_last, nf * mult, kernel_size=kernel,
                                    stride=stride, padding=pad, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(nf * mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(nf * mult, 1, kernel_size=kernel, stride=1, padding=pad))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)