"""
Pérdida Perceptual Multiescala adaptada para matrices CSI de GIROS (2, 544, 14).
- 4 etapas de reducción asimétrica en frecuencia (stride=(2, 1), kernel=(3, 1)).
- Mantiene invariable la dimensión temporal de 14 símbolos OFDM.
- Parámetros entrenables conjuntamente con el generador sin pesos preentrenados ajenos.
"""

import torch
import torch.nn as nn
from collections import namedtuple


class CSIPerceptualNetwork(nn.Module):
    """
    Extractor multiescala de 4 etapas para matrices CSI (2, 544, 14).
    Aplica strides asimétricos (2, 1) y kernels (3, 1) para evaluar las
    subportadoras sin alterar nunca los 14 símbolos OFDM.
    """
    def __init__(self, in_channels=2):
        super(CSIPerceptualNetwork, self).__init__()

        
        self.slice1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        
        self.slice2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        
        self.slice3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        
        self.slice4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        outputs = namedtuple("Outputs", ['stage1', 'stage2', 'stage3', 'stage4'])
        return outputs(h1, h2, h3, h4)


class NetLinLayer(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(NetLinLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )
        # Inicialización con pesos positivos suaves para evitar que reste pérdida al inicio
        nn.init.normal_(self.model[1].weight, mean=0.1, std=0.01)


class LPIPS(nn.Module):
    def __init__(self, in_channels=2):
        super(LPIPS, self).__init__()
        self.channels = [64, 128, 256, 512]
        self.net = CSIPerceptualNetwork(in_channels=in_channels)
        self.lins = nn.ModuleList([
            NetLinLayer(self.channels[0]),
            NetLinLayer(self.channels[1]),
            NetLinLayer(self.channels[2]),
            NetLinLayer(self.channels[3])
        ])

        
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, real_x, fake_x):
        feats_real = self.net(real_x)
        feats_fake = self.net(fake_x)

        loss = 0.0
        for i in range(len(self.channels)):
            diff = (norm_tensor(feats_real[i]) - norm_tensor(feats_fake[i])) ** 2
            loss = loss + spatial_average(self.lins[i].model(diff)).mean()

        return loss


def norm_tensor(x):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + 1e-10)


def spatial_average(x):
    return x.mean([2, 3], keepdim=True)
