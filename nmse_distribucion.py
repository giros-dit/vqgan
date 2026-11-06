#!/usr/bin/env python3
"""
Distribución del NMSE en el test split a partir de reconstrucciones ya guardadas
(sirve para el VQGAN y para el VQ-VAE: solo hay que darle la carpeta).

Uso:
  python nmse_distribucion.py --dataset GIROS --recon_dir <carpeta con los .pt reconstruidos>
"""

import argparse
import os
import numpy as np
import torch

from utils import DATASETS, get_spec, split_dataset


def load_chw(path):
    t = torch.load(path, map_location="cpu", weights_only=False).float()
    if t.ndim == 4:
        t = t.squeeze(0)
    if t.ndim == 2:
        t = t.unsqueeze(0)
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=DATASETS.keys())
    p.add_argument("--recon_dir", required=True)
    args = p.parse_args()

    spec = get_spec(args.dataset)
    ds = spec["make_dataset"](spec["root_dir"])
    _, _, test = split_dataset(ds)
    files = [ds.tensor_files[i] for i in test.indices]

    ratios, errs, pows = [], [], []
    for f in files:
        r_path = os.path.join(args.recon_dir, os.path.basename(f))
        if not os.path.isfile(r_path):
            continue
        o = load_chw(f)
        if "channel_idx" in spec:
            o = o[spec["channel_idx"]]
        r = load_chw(r_path)
        c = min(o.shape[0], r.shape[0])
        h = min(spec["height"], o.shape[-2], r.shape[-2])
        w = min(spec["width"], o.shape[-1], r.shape[-1])
        o, r = o[:c, :h, :w], r[:c, :h, :w]
        err = torch.sum((o - r) ** 2).item()
        pw = torch.sum(o ** 2).item()
        if pw > 0:
            ratios.append(err / pw)
            errs.append(err)
            pows.append(pw)

    ratios, errs, pows = np.array(ratios), np.array(errs), np.array(pows)
    db = 10 * np.log10(ratios)
    print(f"Grids de test evaluadas: {len(ratios)}")
    print(f"NMSE medio (media lineal, como el compressor): {10 * np.log10(ratios.mean()):6.2f} dB")
    print(f"NMSE global (error total / potencia total):    {10 * np.log10(errs.sum() / pows.sum()):6.2f} dB")
    print(f"Mediana del NMSE por grid:                     {np.median(db):6.2f} dB")
    print(f"Percentiles 10 / 25 / 75 / 90:  {np.percentile(db, 10):.2f} / {np.percentile(db, 25):.2f} / "
          f"{np.percentile(db, 75):.2f} / {np.percentile(db, 90):.2f} dB")
    print(f"Grids peor que -3 dB: {100 * np.mean(db > -3):.1f} %")

    # Las de más potencia frente a las de menos
    orden = np.argsort(pows)
    mitad = len(orden) // 2
    baja, alta = orden[:mitad], orden[mitad:]
    print(f"Mitad con MENOS potencia: NMSE medio {10 * np.log10(ratios[baja].mean()):.2f} dB")
    print(f"Mitad con MÁS potencia:   NMSE medio {10 * np.log10(ratios[alta].mean()):.2f} dB")


if __name__ == "__main__":
    main()