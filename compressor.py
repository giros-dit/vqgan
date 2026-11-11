import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from utils import DATASETS, get_spec, models_dir, recon_dir, fit_to_spec
from vqgan import VQGAN


class RawGridDataset(Dataset):
    """
    Lee cada fichero una sola vez y lo prepara como el compressor del VQ-VAE:
    recorta a las filas reales, ajusta las columnas y rellena hasta el tamaño con padding.
    Devuelve también el índice y la forma original para guardar la reconstrucción igual.
    """

    def __init__(self, files, spec):
        self.files = files
        self.spec = spec

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(raw, torch.Tensor):
            raise TypeError(f"{os.path.basename(path)} no es un tensor: {type(raw).__name__}")
        orig_ndim = raw.ndim
        x = raw.to(torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        w_orig = min(x.shape[-1], self.spec["width"])
        x = fit_to_spec(x.unsqueeze(0), self.spec).squeeze(0)
        return x, idx, orig_ndim, w_orig


def test_split_indices(n, seed=42):
    """Reproduce el 10 % de test de random_split([0.8, 0.1, 0.1]) con la misma semilla."""
    _, _, test_idx = torch.utils.data.random_split(
        range(n), [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(seed))
    return set(test_idx)


def main():
    p = argparse.ArgumentParser(description="Compresión y descompresión con el VQGAN")
    p.add_argument("--dataset", type=str, default="GIROS", choices=DATASETS.keys())
    p.add_argument("--quantizer", type=str, default="ema", choices=["ema", "original"])
    p.add_argument("--n_embeddings", type=int, default=512,
                   help="Solo se usa para las rutas por defecto (nembeddings{K}, _{K})")
    p.add_argument("--downsample", type=int, default=16, choices=[4, 16],
                   help="Reducción con la que se entrenó el modelo (elige las carpetas por defecto)")
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_threads", type=int, default=8,
                   help="Hilos para escribir los ficheros en paralelo")
    p.add_argument("--solo_test", action="store_true",
                   help="Procesa solo el test split (más las --extra_grids)")
    p.add_argument("--extra_grids", type=str, nargs="*", default=[],
                   help="Nombres (sin .pt) que se procesan siempre con --solo_test, p. ej. la grid del mosaico")
    args = p.parse_args()

    spec = get_spec(args.dataset, args.downsample)
    k = args.n_embeddings
    if args.model_path is None:
        args.model_path = os.path.join(models_dir(args.dataset, args.quantizer, args.downsample),
                                       f"nembeddings{k}", "model.pth")
    if args.output_dir is None:
        args.output_dir = os.path.join(recon_dir(args.dataset, args.quantizer, args.downsample),
                                       f"torch.float8_e4m3fn_{k}")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Dispositivo: {device}")
    print(f"--> Modelo: {args.model_path}")
    print(f"--> Salida: {args.output_dir}")

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    hp = checkpoint["hyperparameters"]
    if hp.get("dataset") != args.dataset:
        print(f"[AVISO] El modelo se entrenó con {hp.get('dataset')} y estás usando {args.dataset}")
    if hp.get("downsample", 16) != args.downsample:
        raise ValueError(f"El modelo se entrenó con --downsample {hp.get('downsample', 16)}, "
                         f"pero has puesto --downsample {args.downsample}")
    if hp.get("down2d", False) != spec["down2d"]:
        raise ValueError(f"El modelo se entrenó con down2d = {hp.get('down2d', False)}, "
                         f"pero el dataset {args.dataset} tiene down2d = {spec['down2d']}")

    model = VQGAN(SimpleNamespace(**hp)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    K = hp["num_codebook_vectors"]
    tokens = (spec["padded_height"] // spec["f_h"]) * (spec["padded_width"] // spec["f_w"])
    values = spec["channels"] * spec["height"] * spec["width"]
    reduccion = f"{spec['f_h']} x {spec['f_w']}"
    print(f"[OK] Modelo cargado (K = {K}, dimensión = {hp['latent_dim']}, "
          f"cuantizador = {hp['quantizer']}, reducción x{spec['downsample']} = {reduccion})")
    print(f"--> {tokens} tokens por grid | {tokens * np.log2(K) / values:.4f} bits por valor")

    # Misma lista de ficheros (y mismo orden) que en el entrenamiento
    files = spec["make_dataset"](spec["root_dir"]).tensor_files
    test_idx = test_split_indices(len(files))

    if args.solo_test:
        extra = set(args.extra_grids)
        extra_idx = {i for i, f in enumerate(files) if os.path.basename(f)[:-3] in extra}
        if len(extra_idx) < len(extra):
            print(f"[AVISO] Solo se han encontrado {len(extra_idx)} de las {len(extra)} --extra_grids")
        selected = sorted(test_idx | extra_idx)
        print(f"--> Solo test: {len(test_idx)} grids de test + {len(extra_idx - test_idx)} extra")
    else:
        selected = list(range(len(files)))
        print(f"--> Procesando {len(files)} grids ({len(test_idx)} del test split)...")

    # Subset pasa al dataset el índice ORIGINAL, así que idxs sigue apuntando a files
    loader = DataLoader(Subset(RawGridDataset(files, spec), selected),
                        batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    h, W = spec["height"], spec["width"]
    nmse_all, nmse_test = [], []
    pool = ThreadPoolExecutor(max_workers=args.save_threads)

    with torch.no_grad():
        pbar = tqdm(loader, desc="Comprimiendo", unit="lote", dynamic_ncols=True)
        for x, idxs, ndims, w_origs in pbar:
            recon = model.reconstruct(x.to(device)).cpu()                  # (B, C, h, W)
            out = torch.clamp(recon, -448.0, 448.0).to(torch.float8_e4m3fn)

            futures = []
            for j in range(len(idxs)):
                i = idxs[j].item()
                w = w_origs[j].item()
                left = (W - w) // 2          # columnas replicadas a la izquierda (solo FAWN, si las hubo)

                o = out[j, :, :, left:left + w].contiguous().clone()      # clone: guarda solo este grid
                orig = x[j, :, :h, left:left + w]
                save = o.squeeze(0) if ndims[j].item() == 2 else o
                futures.append(pool.submit(
                    torch.save, save, os.path.join(args.output_dir, os.path.basename(files[i]))))

                rec = o.to(torch.float32)
                power = torch.sum(orig ** 2).item()
                if power > 0:
                    ratio = torch.sum((orig - rec) ** 2).item() / power
                    nmse_all.append(ratio)
                    if i in test_idx:
                        nmse_test.append(ratio)

            for f in futures:                # espera a que se escriba el lote (y saca errores)
                f.result()

            if nmse_all:
                pbar.set_postfix(nmse_dB=f"{10 * np.log10(np.mean(nmse_all)):.2f}")

    pool.shutdown()

    def to_db(v):
        return 10 * np.log10(np.mean(v)) if v else float("nan")

    print(f"\n[ÉXITO] Grids guardados en:\n{args.output_dir}")
    if not args.solo_test:
        print(f"\nNMSE medio (todos los grids): {to_db(nmse_all):.2f} dB")
    print(f"NMSE medio (solo test split):  {to_db(nmse_test):.2f} dB  <- este es el de la curva")


if __name__ == "__main__":
    main()