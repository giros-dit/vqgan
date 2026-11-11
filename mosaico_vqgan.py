#!/usr/bin/env python3
"""
Mosaico de reconstrucciones del VQGAN: la muestra original y sus reconstrucciones
con varios K, con la misma escala de colores en todos los paneles (la del original).

Uso:
  python mosaico_vqgan.py --dataset FAWN --grid_name <nombre>
  python mosaico_vqgan.py --dataset FAWN --grid_name <nombre> --downsample 4
Guarda por defecto en RESULTADOS/<DATASET>/ (x16) o RESULTADOSx4/<DATASET>/ (x4)
con el nombre mosaico_vqgan_<dataset>_<q>.pdf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import DATASETS, get_spec, recon_dir

# Etiquetas de los ejes (las mismas que en el mosaico del VQ-VAE)
AXIS_LABELS = {
    "FAWN": ("Subportadora", "Símbolo OFDM"),
    "GIROS": ("Subportadora", "Símbolo OFDM"),
    "GIROS_AMP": ("Subportadora", "Símbolo OFDM"),
    "CAEZ": ("Frecuencia (fila)", "Columna"),
    "FRAUNHOFER": ("Retardo (muestra de la CIR)", "Estación base"),
}


def load_chw(path):
    """Carga un tensor y lo devuelve como (C, H, W) en float32."""
    t = torch.load(path, map_location="cpu", weights_only=False).float()
    if t.ndim == 4:
        t = t.squeeze(0)
    if t.ndim == 2:
        t = t.unsqueeze(0)
    return t


def to_image(t):
    """2 canales -> sqrt(c0^2 + c1^2) (igual que en el VQ-VAE); 1 canal -> tal cual."""
    if t.shape[0] == 2:
        return torch.sqrt(t[0] ** 2 + t[1] ** 2).numpy()
    return t[0].numpy()


def nmse_db(orig, recon):
    mse = torch.mean((orig - recon) ** 2).item()
    power = torch.mean(orig ** 2).item()
    if mse <= 0 or power <= 0:
        return float("nan")
    return 10 * np.log10(mse / power)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=DATASETS.keys())
    p.add_argument("--quantizer", default="ema", choices=["ema", "original"])
    p.add_argument("--downsample", type=int, default=16, choices=[4, 16],
                   help="Reducción del VQGAN con que se hicieron las reconstrucciones")
    p.add_argument("--grid_name", required=True, help="Nombre del fichero (con o sin .pt)")
    p.add_argument("--ks", type=int, nargs="+", default=[512, 128, 32, 8, 2])
    p.add_argument("--output", default=None)
    args = p.parse_args()

    spec = get_spec(args.dataset)
    ylabel, xlabel = AXIS_LABELS.get(args.dataset, ("Fila", "Columna"))
    name = args.grid_name[:-3] if args.grid_name.endswith(".pt") else args.grid_name
    recon_base = recon_dir(args.dataset, args.quantizer, args.downsample)
    carpeta = "RESULTADOS" if args.downsample == 16 else f"RESULTADOSx{args.downsample}"

    orig = load_chw(os.path.join(spec["root_dir"], f"{name}.pt"))
    if "channel_idx" in spec:                      # GIROS_AMP: solo el módulo
        orig = orig[spec["channel_idx"]]
    orig = orig[:, :spec["height"], :spec["width"]]

    panels = [("Original", to_image(orig))]
    for k in args.ks:
        f = os.path.join(recon_base, f"torch.float8_e4m3fn_{k}", f"{name}.pt")
        if not os.path.isfile(f):
            print(f"[AVISO] No existe la reconstrucción con K={k}: {f}")
            continue
        recon = load_chw(f)
        h = min(orig.shape[-2], recon.shape[-2])
        w = min(orig.shape[-1], recon.shape[-1])
        o, r = orig[:, :h, :w], recon[:, :h, :w]
        n = nmse_db(o, r)
        print(f"K={k}: NMSE de esta muestra = {n:.2f} dB")
        panels.append((f"$K = {k}$\n{n:.1f} dB", to_image(r)))

    vmin = panels[0][1].min()
    vmax = panels[0][1].max()

    fig, axes = plt.subplots(1, len(panels), figsize=(1.9 * len(panels) + 1, 4.8),
                             sharey=True, constrained_layout=True)
    for ax, (title, img) in zip(axes, panels):
        im = ax.imshow(img, aspect="auto", cmap="viridis", interpolation="nearest",
                       vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(ylabel, fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.9, label="Módulo", extend="both")

    out = args.output or os.path.join(
        carpeta, args.dataset,
        f"mosaico_vqgan_{args.dataset.lower()}_{args.quantizer}.pdf")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"--> Guardado en: {out}")


if __name__ == "__main__":
    main()