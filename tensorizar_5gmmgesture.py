"""Tensoriza 5GmmGesture: una grid (1, 50, 56) por fichero, en float8_e4m3fn, usuarios 1 a 8 en orden."""
import argparse
import os

import torch

BASE = '/data' if os.path.isdir('/data/datasets') else '/home/alfonso.inigo/disco6g-data'

p = argparse.ArgumentParser()
p.add_argument('--data_dir', default=os.path.join(BASE, 'datasets', '5GmmGesture'))
p.add_argument('--out_dir', default=os.path.join(BASE, 'datasets', '5GmmGesture', 'tensored', 'torch.float8_e4m3fn'))
p.add_argument('--usuarios', type=int, nargs='+', default=list(range(1, 9)))
p.add_argument('--max_ejemplos', type=int, default=None, help='Solo para pruebas: ejemplos por usuario')
args = p.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
total = 0
for u in args.usuarios:
    datos = torch.load(os.path.join(args.data_dir, f'PPBP_data_user{u}.pth'), map_location='cpu', weights_only=False)
    etiq = torch.load(os.path.join(args.data_dir, f'labels_user{u}.pth'), map_location='cpu', weights_only=False)
    n = datos.shape[0] if args.max_ejemplos is None else min(args.max_ejemplos, datos.shape[0])
    print(f'Usuario {u}: {tuple(datos.shape)} -> guardo {n} ejemplos x {datos.shape[1]} instantes', flush=True)

    datos = datos[:n].to(torch.float8_e4m3fn)          # (n, 20, 50, 56) en float8
    for e in range(n):
        g = int(etiq[e])
        for t in range(datos.shape[1]):
            # .clone() para que cada fichero guarde solo su grid y no todo el tensor
            torch.save(datos[e, t:t + 1].clone(),
                       os.path.join(args.out_dir, f'PPBP-u{u}-e{e:04d}-g{g}-t{t:02d}.pt'))
            total += 1
        if (e + 1) % 500 == 0:
            print(f'  {e + 1}/{n}', flush=True)
    del datos, etiq

print(f'Hecho: {total} grids en {args.out_dir}')