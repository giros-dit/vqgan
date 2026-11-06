#!/usr/bin/env python3
"""
Lista las grids de test de GIROS ordenadas por NMSE, separando el error del
módulo (canal 0) y de la fase (canal 1), para elegir cuáles dibujar.

Uso:
  python grids_giros.py --recon_dir <carpeta con las reconstrucciones> [--grid <nombre>]
"""

import argparse
import os
import numpy as np
import torch

from utils import get_spec, split_dataset


def load_chw(path):
    t = torch.load(path, map_location="cpu", weights_only=False).float()
    if t.ndim == 4:
        t = t.squeeze(0)
    if t.ndim == 2:
        t = t.unsqueeze(0)
    return t


def db(x):
    return 10 * np.log10(x) if x > 0 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="GIROS")
    p.add_argument("--recon_dir", required=True)
    p.add_argument("--n", type=int, default=5, help="Cuántas grids enseñar de cada extremo")
    p.add_argument("--grid", default=None, help="Dice si esta grid es de train, val o test")
    args = p.parse_args()

    spec = get_spec(args.dataset)
    ds = spec["make_dataset"](spec["root_dir"])
    tr, va, te = split_dataset(ds)
    nombres = [os.path.basename(f)[:-3] for f in ds.tensor_files]

    if args.grid:
        i = nombres.index(args.grid)
        parte = "TRAIN" if i in tr.indices else "VAL" if i in va.indices else "TEST"
        print(f"{args.grid} -> {parte}\n")

    filas = []
    for i in te.indices:
        f = ds.tensor_files[i]
        r_path = os.path.join(args.recon_dir, os.path.basename(f))
        if not os.path.isfile(r_path):
            continue
        o = load_chw(f)[:, :spec["height"], :spec["width"]]
        r = load_chw(r_path)[:, :spec["height"], :spec["width"]]
        e = (o - r) ** 2
        p_tot = (o ** 2).sum().item()
        filas.append(dict(
            nombre=nombres[i],
            nmse=e.sum().item() / p_tot,
            nmse_mod=e[0].sum().item() / (o[0] ** 2).sum().item(),
            nmse_fase=e[1].sum().item() / (o[1] ** 2).sum().item(),
            peso_fase=(o[1] ** 2).sum().item() / p_tot,     # parte de la potencia que es fase
            modulo_medio=o[0].mean().item(),
        ))

    if not filas:
        print(f"[ERROR] No hay reconstrucciones del test en {args.recon_dir}")
        return

    filas.sort(key=lambda d: d["nmse"])
    cab = f"{'grid':<28} {'NMSE':>7} {'módulo':>8} {'fase':>7} {'% fase':>7} {'|H| medio':>9}"

    def imprime(lista):
        print(cab)
        for d in lista:
            print(f"{d['nombre']:<28} {db(d['nmse']):7.2f} {db(d['nmse_mod']):8.2f} "
                  f"{db(d['nmse_fase']):7.2f} {100 * d['peso_fase']:6.1f}% {d['modulo_medio']:9.2f}")

    print(f"Grids de test: {len(filas)} (NMSE en dB; '% fase' = parte de la potencia que es fase)\n")
    print(f"=== LAS {args.n} MEJORES ===")
    imprime(filas[:args.n])
    m = len(filas) // 2
    print(f"\n=== {args.n} EN LA MEDIANA ===")
    imprime(filas[m - args.n // 2: m - args.n // 2 + args.n])
    print(f"\n=== LAS {args.n} PEORES ===")
    imprime(filas[-args.n:])


if __name__ == "__main__":
    main()