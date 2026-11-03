"""
Vector Quantizer (Codebook) para matrices CSI de GIROS.
Cuantiza el mapa latente (B, latent_dim, 34, 14) al espacio discreto de 1024 vectores.
Incluye cálculo de perplejidad para monitorizar el uso efectivo de los vectores.
"""

import torch
import torch.nn as nn


class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        b, c, h, w = z.shape
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        d = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2)

        indices_grid = min_encoding_indices.view(b, h, w)

        return z_q, indices_grid, loss

    def calculate_perplexity(self, min_encoding_indices):
       
        flat_indices = min_encoding_indices.view(-1)
        encodings = torch.zeros(flat_indices.shape[0], self.num_codebook_vectors, device=flat_indices.device)
        encodings.scatter_(1, flat_indices.unsqueeze(1), 1)

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity.item()