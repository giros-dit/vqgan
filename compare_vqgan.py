#!/usr/bin/env python3
"""
Original frente a reconstrucción del VQGAN para una muestra.

Uso:
  python compare_vqgan.py --dataset FAWN --n_embeddings 512 --grid_name <nombre>
  python compare_vqgan.py --dataset FAWN --n_embeddings 512 --grid_name <nombre> --downsample 4
Guarda por defecto en RESULTADOS/<DATASET>/ (x16) o RESULTADOSx4/<DATASET>/ (x4)
con el nombre compare_vqgan_<dataset>_K<K>_<grid>.pdf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import DATASETS, get_spec, recon_dir


def load_chw(path):
    """Carga un tensor y lo devuelve como (C, H, W) en float32."""
    t = torch.load(path, map_location="cpu", weights_only=False).float()
    if t.ndim == 4:
        t = t.squeeze(0)
    if t.ndim == 2:
        t = t.unsqueeze(0)
    return t


def plot_comparison(orig_path, recon_path, title_extra="", save_path=None):
    t_orig = load_chw(orig_path)
    t_recon = load_chw(recon_path)

    # Zona común (por si alguno conserva filas de padding o columnas de más)
    h = min(t_orig.shape[-2], t_recon.shape[-2])
    w = min(t_orig.shape[-1], t_recon.shape[-1])
    c = min(t_orig.shape[0], t_recon.shape[0])
    t_orig = t_orig[:c, :h, :w]
    t_recon = t_recon[:c, :h, :w]

    # NMSE de esta muestra
    mse = torch.mean((t_orig - t_recon) ** 2).item()
    power = torch.mean(t_orig ** 2).item()
    nmse_db = 10 * np.log10(mse / power) if (power > 0 and mse > 0) else float("nan")

    if t_orig.shape[0] == 2:
        orig_img = torch.sqrt(t_orig[0] ** 2 + t_orig[1] ** 2).numpy()
        recon_img = torch.sqrt(t_recon[0] ** 2 + t_recon[1] ** 2).numpy()
        value_label = "|I + jQ|"
    else:
        orig_img = t_orig[0].numpy()
        recon_img = t_recon[0].numpy()
        value_label = "canal 0"

    vmin = min(orig_img.min(), recon_img.min())
    vmax = max(orig_img.max(), recon_img.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)
    for ax, img, title in zip(
        axes, [orig_img, recon_img],
        [f"Original ({value_label})\n{os.path.basename(orig_path)}",
         f"Reconstruido VQGAN{title_extra} (NMSE: {nmse_db:.2f} dB)\n{os.path.basename(recon_path)}"],
    ):
        im = ax.imshow(img, interpolation="nearest", aspect="auto", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(f"Symbol (0 a {w - 1})")
    axes[0].set_ylabel(f"Subcarrier (0 a {h - 1})")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)

    print(f"Muestra: {os.path.basename(orig_path)}")
    print(f"NMSE: {nmse_db:.2f} dB | MSE: {mse:.6e}")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"--> Imagen guardada en: {save_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Comparar original frente a reconstrucción del VQGAN")
    p.add_argument("--dataset", required=True, choices=DATASETS.keys())
    p.add_argument("--quantizer", default="ema", choices=["ema", "original"])
    p.add_argument("--downsample", type=int, default=16, choices=[4, 16],
                   help="Reducción del VQGAN con que se hicieron las reconstrucciones")
    p.add_argument("--n_embeddings", type=int, default=512)
    p.add_argument("--grid_name", required=True, help="Nombre del fichero (con o sin .pt)")
    p.add_argument("--recon_dir", default=None,
                   help="Carpeta de reconstrucciones (por defecto: <dataset>/vqgand_{q}[_x4]/torch.float8_e4m3fn_{K})")
    p.add_argument("--output", default=None,
                   help="Por defecto: RESULTADOS[x4]/<DATASET>/compare_vqgan_<dataset>_K<K>_<grid>.pdf")
    args = p.parse_args()

    spec = get_spec(args.dataset)
    name = args.grid_name[:-3] if args.grid_name.endswith(".pt") else args.grid_name
    k = args.n_embeddings
    carpeta = "RESULTADOS" if args.downsample == 16 else f"RESULTADOSx{args.downsample}"

    if args.recon_dir is None:
        args.recon_dir = os.path.join(recon_dir(args.dataset, args.quantizer, args.downsample),
                                      f"torch.float8_e4m3fn_{k}")
    if args.output is None:
        args.output = os.path.join(carpeta, args.dataset,
                                   f"compare_vqgan_{args.dataset.lower()}_K{k}_{name}.pdf")

    orig_file = os.path.join(spec["root_dir"], f"{name}.pt")
    recon_file = os.path.join(args.recon_dir, f"{name}.pt")
    if not os.path.isfile(orig_file):
        print(f"Error: no existe el original en: {orig_file}")
        return
    if not os.path.isfile(recon_file):
        print(f"Error: no existe la reconstrucción en: {recon_file}")
        return

    plot_comparison(orig_file, recon_file, title_extra=f" x{args.downsample} K={k}",
                    save_path=args.output)


if __name__ == "__main__":
    main()