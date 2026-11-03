import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        # Mismo dispositivo que la entrada (funciona igual en CPU y GPU)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


class VectorQuantizerEMA(nn.Module):
    """
    Cuantizador con codebook actualizado por media móvil (EMA) y reinicio de
    códigos muertos. Misma interfaz que VectorQuantizer:
        loss, z_q, perplexity, min_encodings, min_encoding_indices = vq(z)

    Diferencias con el original:
      - Los vectores del codebook NO se aprenden por gradiente: cada vector se
        mueve hacia la media de las salidas del encoder que tiene asignadas.
      - El codebook se inicializa con salidas reales del encoder (primer batch),
        no con valores minúsculos en ±1/n_e.
      - Los códigos que dejan de usarse (cluster_size < dead_threshold) se
        reinician con salidas del encoder del batch actual.
      - La loss solo lleva el término de compromiso: beta * ||z_e - sg(e)||^2.
    """

    def __init__(self, n_e, e_dim, beta, decay=0.99, eps=1e-5, dead_threshold=1.0):
        super(VectorQuantizerEMA, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.dead_threshold = dead_threshold

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.embedding.weight.requires_grad_(False)  # se actualiza a mano

        # Estado de la EMA (se guarda en el state_dict y viaja con el modelo)
        self.register_buffer("cluster_size", torch.zeros(n_e))
        self.register_buffer("embed_avg", self.embedding.weight.data.clone())
        self.register_buffer("initted", torch.tensor(False))

    @staticmethod
    def _sample_vectors(z_flat, n):
        """n vectores al azar del batch (con reemplazo si hay menos que n)."""
        idx = torch.randint(0, z_flat.shape[0], (n,), device=z_flat.device)
        return z_flat[idx]

    @torch.no_grad()
    def _init_codebook(self, z_flat):
        init = self._sample_vectors(z_flat, self.n_e)
        self.embedding.weight.data.copy_(init)
        self.embed_avg.copy_(init)
        self.cluster_size.fill_(1.0)
        self.initted.fill_(True)

    @torch.no_grad()
    def _ema_update(self, z_flat, min_encodings):
        counts = min_encodings.sum(0)                       # (n_e,)
        embed_sum = min_encodings.t() @ z_flat              # (n_e, e_dim)

        self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        # Suavizado de Laplace para no dividir entre 0
        total = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (total + self.n_e * self.eps) * total
        self.embedding.weight.data.copy_(self.embed_avg / smoothed.unsqueeze(1))

        # Reinicio de códigos muertos
        dead = self.cluster_size < self.dead_threshold
        n_dead = int(dead.sum().item())
        if n_dead > 0:
            new = self._sample_vectors(z_flat, n_dead)
            self.embedding.weight.data[dead] = new
            self.embed_avg[dead] = new
            self.cluster_size[dead] = 1.0

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        if self.training and not bool(self.initted):
            self._init_codebook(z_flattened.detach())

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        if self.training:
            self._ema_update(z_flattened.detach(), min_encodings)

        # Solo término de compromiso: el codebook ya se mueve por EMA
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2)

        # Estimador straight-through
        z_q = z + (z_q - z).detach()

        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices
