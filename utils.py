import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    FAWN_DIR, GIROS_DIR, CAEZ_DIR, FRAUNHOFER_DIR, MMGESTURE_DIR,
    FAWN_ROOT_DIR, GIROS_ROOT_DIR, CAEZ_ROOT_DIR, FRAUNHOFER_ROOT_DIR, MMGESTURE_ROOT_DIR,
    VQGAN_MODELS_SUBDIR, VQGAN_RECON_SUBDIR,
)
from grid_tensor_fawn_dataset import FAWNGridSymbolTensorDataset
from grid_tensor_giros_dataset import GIROSTensorDataset
from grid_tensor_caez_dataset import CAEZGridTensorDataset
from grid_tensor_fraunhofer_dataset import FraunhoferTensorDataset
from grid_tensor_5gmmgesture_dataset import MMGestureTensorDataset

# height = filas reales (eje que se reduce), width = columnas reales.
# down2d = True: la reducción se reparte entre los dos ejes (solo 5GmmGesture).
# El padding (padded_height, padded_width) lo calcula get_spec según --downsample.
DATASETS = {
    "FAWN": dict(
        channels=2, height=360, width=6,
        base_dir=FAWN_DIR, root_dir=FAWN_ROOT_DIR,
        make_dataset=FAWNGridSymbolTensorDataset,
    ),
    "GIROS": dict(
        channels=2, height=540, width=14,
        base_dir=GIROS_DIR, root_dir=GIROS_ROOT_DIR,
        make_dataset=GIROSTensorDataset,
    ),
    # GIROS solo con el canal de módulo (el canal 1 es la fase, aleatoria)
    "GIROS_AMP": dict(
        channels=1, height=540, width=14,
        base_dir=GIROS_DIR, root_dir=GIROS_ROOT_DIR,
        make_dataset=GIROSTensorDataset,
        channel_idx=[0], tag="_amp",
    ),
    "CAEZ": dict(
        channels=1, height=273, width=10,
        base_dir=CAEZ_DIR, root_dir=CAEZ_ROOT_DIR,
        make_dataset=CAEZGridTensorDataset,
    ),
    "FRAUNHOFER": dict(
        channels=1, height=49, width=6,
        base_dir=FRAUNHOFER_DIR, root_dir=FRAUNHOFER_ROOT_DIR,
        make_dataset=FraunhoferTensorDataset,
    ),
    # 5GmmGesture: haces Tx (filas) x haces Rx (columnas), reducción en los dos ejes
    "5GMMGESTURE": dict(
        channels=1, height=50, width=56,
        base_dir=MMGESTURE_DIR, root_dir=MMGESTURE_ROOT_DIR,
        make_dataset=MMGestureTensorDataset,
        down2d=True,
    ),
}


def get_spec(dataset_name, downsample=16):
    """
    Datos del dataset con el padding calculado para la reducción del encoder.
      - Normal: factor f_h = downsample en vertical, f_w = 1 en horizontal.
      - down2d: f_h = f_w = sqrt(downsample) (x16 -> 4 x 4, x4 -> 2 x 2).
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' no soportado. Opciones: {list(DATASETS)}")
    spec = dict(DATASETS[dataset_name])
    spec["downsample"] = downsample
    spec["down2d"] = spec.get("down2d", False)
    if spec["down2d"]:
        f = int(round(math.sqrt(downsample)))
        if f * f != downsample:
            raise ValueError(f"Con reducción en los dos ejes, downsample debe ser 4 o 16, no {downsample}")
        spec["f_h"], spec["f_w"] = f, f
    else:
        spec["f_h"], spec["f_w"] = downsample, 1
    spec["padded_height"] = math.ceil(spec["height"] / spec["f_h"]) * spec["f_h"]
    spec["padded_width"] = math.ceil(spec["width"] / spec["f_w"]) * spec["f_w"]
    return spec


def _suffix(spec):
    """'_amp' para GIROS_AMP y '_x4' si la reducción no es la original (16)."""
    s = spec.get("tag", "")
    if spec["downsample"] != 16:
        s += f"_x{spec['downsample']}"
    return s


def models_dir(dataset_name, quantizer, downsample=16):
    """<dataset>/vqgan-models-{q}{sufijo}"""
    spec = get_spec(dataset_name, downsample)
    return os.path.join(spec["base_dir"], VQGAN_MODELS_SUBDIR.format(q=quantizer) + _suffix(spec))


def recon_dir(dataset_name, quantizer, downsample=16):
    """<dataset>/vqgand_{q}{sufijo}"""
    spec = get_spec(dataset_name, downsample)
    return os.path.join(spec["base_dir"], VQGAN_RECON_SUBDIR.format(q=quantizer) + _suffix(spec))


def fit_to_spec(x, spec):
    """
    (B, C, H, W) -> (B, C', padded_height, padded_width):
      0. Si el dataset lo indica, se queda solo con algunos canales (GIROS_AMP: el módulo).
      1. Recorta a las filas reales (quita el padding que traiga el dataset del VQ-VAE).
      2. Columnas:
         - down2d: recorta a las columnas reales y rellena con ceros hasta padded_width;
         - normal: si faltan, replica los bordes (como en FAWN); si sobran, recorta.
      3. Rellena con ceros al final del eje vertical hasta padded_height.
    """
    if "channel_idx" in spec:
        x = x[:, spec["channel_idx"], :, :]
    x = x[:, :, :spec["height"], :]

    if spec.get("down2d", False):
        x = x[..., :spec["width"]]
        pad_w = spec["padded_width"] - x.shape[-1]
        if pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, 0), mode="constant", value=0.0)
    else:
        w = x.shape[-1]
        if w < spec["width"]:
            diff = spec["width"] - w
            x = F.pad(x, (diff // 2, diff - diff // 2, 0, 0), mode="replicate")
        elif w > spec["width"]:
            x = x[..., :spec["width"]]

    pad_h = spec["padded_height"] - x.shape[-2]
    if pad_h > 0:
        x = F.pad(x, (0, 0, 0, pad_h), mode="constant", value=0.0)
    return x


def make_collate_fn(spec):
    def collate(batch):
        tensors, names = zip(*batch)
        x = torch.stack([t.float() if t.ndim == 3 else t.float().unsqueeze(0) for t in tensors], dim=0)
        return fit_to_spec(x, spec), list(names)
    return collate


def split_dataset(dataset, seed=42):
    """80 % train, 10 % val, 10 % test con la misma semilla que el VQ-VAE."""
    return torch.utils.data.random_split(
        dataset, [0.8, 0.1, 0.1],
        generator=torch.Generator().manual_seed(seed),
    )


def train_variance(train_data, spec, max_grids=20000, num_workers=0):
    """
    Varianza de los valores del train, sin contar el padding (ni filas ni columnas).
    Como en el VQ-VAE: hasta 20.000 grids elegidos con semilla 0.
    """
    n = min(max_grids, len(train_data))
    idx = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(0))[:n].tolist()
    loader = DataLoader(torch.utils.data.Subset(train_data, idx), batch_size=256,
                        num_workers=num_workers, shuffle=False,
                        collate_fn=make_collate_fn(spec))
    total_sum, total_sq, total_n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x[:, :, :spec["height"], :spec["width"]].double()
            total_sum += x.sum().item()
            total_sq += (x ** 2).sum().item()
            total_n += x.numel()
    mean = total_sum / total_n
    return float(total_sq / total_n - mean ** 2), n


def load_data(args, compute_variance=True):
    """
    Devuelve: train_loader, val_loader, test_data, x_train_var
    Los grids salen como (B, C, padded_height, padded_width) y los nombres como lista.
    """
    spec = get_spec(args.dataset, getattr(args, "downsample", 16))
    dataset = spec["make_dataset"](spec["root_dir"])
    train_data, val_data, test_data = split_dataset(dataset)

    pin_memory = torch.cuda.is_available()
    num_workers = getattr(args, "num_workers", 0)
    collate = make_collate_fn(spec)

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, num_workers=num_workers,
        pin_memory=pin_memory, shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, num_workers=num_workers,
        pin_memory=pin_memory, shuffle=False, collate_fn=collate,
    )
    print(f"--> {args.dataset}: {len(train_data)} train | {len(val_data)} val | {len(test_data)} test")

    x_train_var = None
    if compute_variance:
        x_train_var, n = train_variance(train_data, spec, num_workers=num_workers)
        print(f"--> Varianza de train (con {n} grids): {x_train_var:.6e}")

    return train_loader, val_loader, test_data, x_train_var


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def to_image(t):
    """(C, H, W): 2 canales (I/Q) -> módulo |I + jQ|; 1 canal -> tal cual."""
    if t.shape[0] == 2:
        return torch.sqrt(t[0] ** 2 + t[1] ** 2).numpy()
    return t[0].numpy()


def plot_comparison(real_imgs, recon_imgs, height, save_path=None, width=None):
    """Original frente a reconstrucción de la primera muestra del lote (sin padding)."""
    w = width if width is not None else real_imgs.shape[-1]
    real = to_image(real_imgs[0, :, :height, :w].detach().float().cpu())
    recon = to_image(recon_imgs[0, :, :height, :w].detach().float().cpu())

    vmin, vmax = real.min(), real.max()
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    for ax, img, title in zip(axes, [real, recon], ["Original", "Reconstruido"]):
        im = ax.imshow(img, aspect="auto", cmap="viridis", interpolation="nearest",
                       vmin=vmin, vmax=vmax)
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.8, extend="both")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
