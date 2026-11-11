import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from quantizer import VectorQuantizer, VectorQuantizerEMA


class VQGAN(nn.Module):
    """
    VQGAN para grids de canal: Encoder -> quant_conv -> cuantizador -> post_quant_conv -> Decoder.

    Necesita en args: image_channels, latent_dim, num_codebook_vectors, beta,
    quantizer ("ema" | "original"), height y width (filas y columnas reales, para quitar el padding),
    downsample (16 o 4) y down2d (reducción en los dos ejes, solo 5GmmGesture).

    El cuantizador es EXACTAMENTE el mismo fichero que en el VQ-VAE (quantizer.py).
    """

    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.args = args
        self.height = getattr(args, "height", None)
        self.width = getattr(args, "width", None)
        quantizer = getattr(args, "quantizer", "ema")

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, kernel_size=1)

        if quantizer == "ema":
            self.codebook = VectorQuantizerEMA(
                args.num_codebook_vectors, args.latent_dim, args.beta)
        elif quantizer == "original":
            self.codebook = VectorQuantizer(
                args.num_codebook_vectors, args.latent_dim, args.beta)
        else:
            raise ValueError(f"quantizer debe ser 'ema' u 'original', no '{quantizer}'")

    def quantize(self, z):
        """
        Adaptador: el cuantizador del VQ-VAE devuelve
            (loss, z_q, perplexity, min_encodings, min_encoding_indices)
        y aquí se reordena a (z_q, índices en rejilla (B, H, W), loss, perplexity).
        """
        b, _, h, w = z.shape
        q_loss, z_q, perplexity, _, indices = self.codebook(z)
        return z_q, indices.view(b, h, w), q_loss, perplexity

    def forward(self, imgs):
        """Devuelve: reconstrucción (con padding), índices, q_loss, perplejidad."""
        z = self.quant_conv(self.encoder(imgs))
        z_q, indices, q_loss, perplexity = self.quantize(z)
        decoded = self.decoder(self.post_quant_conv(z_q))
        return decoded, indices, q_loss, perplexity

    def encode(self, imgs):
        """Comprime a índices discretos. Devuelve z_q, índices, q_loss, perplejidad."""
        z = self.quant_conv(self.encoder(imgs))
        return self.quantize(z)

    def decode(self, z_q):
        """Reconstruye a partir del latente cuantizado."""
        return self.decoder(self.post_quant_conv(z_q))

    def reconstruct(self, imgs, crop=True):
        """Reconstrucción; con crop=True quita las filas (y columnas) de padding."""
        decoded, _, _, _ = self.forward(imgs)
        if crop:
            if self.height is not None:
                decoded = decoded[:, :, :self.height, :]
            if self.width is not None:
                decoded = decoded[:, :, :, :self.width]
        return decoded

    def calculate_lambda(self, rec_loss, g_loss):
        """
        λ adaptativo (Esser et al., 2021): iguala lo que empujan la pérdida de
        reconstrucción y la adversarial a la última capa del decoder.
        """
        last_layer_weight = self.decoder.model[-1].weight
        rec_grads = torch.autograd.grad(rec_loss, last_layer_weight, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(rec_grads) / (torch.norm(g_grads) + 1e-4)
        λ = torch.clamp(λ, 0.0, 1e4).detach()
        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, step, threshold, value=0.0):
        """Activa el discriminador solo a partir del paso threshold."""
        if step < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path, device="cpu"):
        """Carga los pesos (mismo formato de checkpoint que el VQ-VAE: clave 'model')."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.load_state_dict(checkpoint["model"])
        else:
            self.load_state_dict(checkpoint)








