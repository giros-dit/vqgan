#!/usr/bin/env python3
"""
Original frente a reconstrucción, canal a canal.
Con 2 canales (GIROS, FAWN): fila de arriba el canal 0 (módulo), fila de abajo el canal 1 (fase).
Con 1 canal (CAEZ, Fraunhofer): una sola fila.

Uso:
  python compare_canales.py --dataset GIROS --n_embeddings 512 --grid_name <nombre>
  python compare_canales.py --dataset GIROS --n_embeddings 512 --grid_name <nombre> --modelo vqvae
Guarda por defecto en RESULTADOS/<DATASET>/canales_<modelo>_<dataset>_K<K>_<grid>.pdf
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


def nmse_db(a, b):
    power = torch.sum(a ** 2).item()
    err = torch.sum((a - b) ** 2).item()
    if power <= 0 or err <= 0:
        return float("nan")
    return 10 * np.log10(err / power)


def plot_canales(orig_path, recon_path, modelo, k, save_path):
    o = load_chw(orig_path)
    r = load_chw(recon_path)
    c = min(o.shape[0], r.shape[0])
    h = min(o.shape[-2], r.shape[-2])
    w = min(o.shape[-1], r.shape[-1])
    o, r = o[:c, :h, :w], r[:c, :h, :w]

    if c == 2:
        canales = [("Módulo (canal 0)", "viridis"), ("Fase (canal 1)", "twilight")]
    else:
        canales = [("Canal 0", "viridis")]

    fig, axes = plt.subplots(c, 2, figsize=(12, 5 * c), sharex=True, sharey=True, squeeze=False)
    for fila, (nombre, cmap) in enumerate(canales):
        vmin, vmax = o[fila].min().item(), o[fila].max().item()   # escala del original
        paneles = [
            (o[fila], f"{nombre} - Original"),
            (r[fila], f"{nombre} - {modelo.upper()} K={k} (NMSE {nmse_db(o[fila], r[fila]):.2f} dB)"),
        ]
        for col, (img, titulo) in enumerate(paneles):
            im = axes[fila, col].imshow(img.numpy(), aspect="auto", cmap=cmap,
                                        interpolation="nearest", vmin=vmin, vmax=vmax)
            axes[fila, col].set_title(titulo, fontsize=9)
        fig.colorbar(im, ax=axes[fila, :].tolist(), shrink=0.8)

    for ax in axes[-1]:
        ax.set_xlabel(f"Columna / símbolo (0 a {w - 1})")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Fila / subportadora (0 a {h - 1})")
    fig.suptitle(f"{os.path.basename(orig_path)} | NMSE total: {nmse_db(o, r):.2f} dB", fontsize=10)

    texto = f"NMSE total: {nmse_db(o, r):.2f} dB"
    for fila, (nombre, _) in enumerate(canales):
        texto += f" | {nombre}: {nmse_db(o[fila], r[fila]):.2f} dB"
    print(f"Muestra: {os.path.basename(orig_path)}")
    print(texto)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"--> Imagen guardada en: {save_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Original frente a reconstrucción, canal a canal")
    p.add_argument("--dataset", required=True, choices=DATASETS.keys())
    p.add_argument("--modelo", default="vqgan", choices=["vqgan", "vqvae"],
                   help="De qué modelo son las reconstrucciones (elige la carpeta por defecto)")
    p.add_argument("--quantizer", default="ema", choices=["ema", "original"])
    p.add_argument("--n_embeddings", type=int, default=512)
    p.add_argument("--grid_name", required=True, help="Nombre del fichero (con o sin .pt)")
    p.add_argument("--recon_dir", default=None,
                   help="Carpeta de reconstrucciones (por defecto: vqgand_{q} o vqvaed_{q} / torch.float8_e4m3fn_{K})")
    p.add_argument("--output", default=None,
                   help="Por defecto: RESULTADOS/<DATASET>/canales_<modelo>_<dataset>_K<K>_<grid>.pdf")
    args = p.parse_args()

    spec = get_spec(args.dataset)
    name = args.grid_name[:-3] if args.grid_name.endswith(".pt") else args.grid_name
    k = args.n_embeddings

    if args.recon_dir is None:
        base = recon_dir(args.dataset, args.quantizer)          # .../vqgand_{q}
        if args.modelo == "vqvae":
            base = base.replace("vqgand_", "vqvaed_")
        args.recon_dir = os.path.join(base, f"torch.float8_e4m3fn_{k}")
    if args.output is None:
        args.output = os.path.join("RESULTADOS", args.dataset,
                                   f"canales_{args.modelo}_{args.dataset.lower()}_K{k}_{name}.pdf")

    orig_file = os.path.join(spec["root_dir"], f"{name}.pt")
    recon_file = os.path.join(args.recon_dir, f"{name}.pt")
    if not os.path.isfile(orig_file):
        print(f"Error: no existe el original en: {orig_file}")
        return
    if not os.path.isfile(recon_file):
        print(f"Error: no existe la reconstrucción en: {recon_file}")
        return

    plot_canales(orig_file, recon_file, args.modelo, k, args.output)


if __name__ == "__main__":
    main()