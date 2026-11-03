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
    FAWN_DIR, GIROS_DIR, CAEZ_DIR, FRAUNHOFER_DIR,
    FAWN_ROOT_DIR, GIROS_ROOT_DIR, CAEZ_ROOT_DIR, FRAUNHOFER_ROOT_DIR,
    VQGAN_MODELS_SUBDIR, VQGAN_RECON_SUBDIR,
)
from grid_tensor_fawn_dataset import FAWNGridSymbolTensorDataset
from grid_tensor_giros_dataset import GIROSTensorDataset
from grid_tensor_caez_dataset import CAEZGridTensorDataset
from grid_tensor_fraunhofer_dataset import FraunhoferTensorDataset

# El encoder del VQGAN reduce x16 en el eje vertical (frecuencia / retardo).
# padded_height = múltiplo de 16 más cercano por encima de height.
DATASETS = {
    "FAWN": dict(
        channels=2, height=360, padded_height=368, width=6,
        base_dir=FAWN_DIR, root_dir=FAWN_ROOT_DIR,
        make_dataset=FAWNGridSymbolTensorDataset,
    ),
    "GIROS": dict(
        channels=2, height=540, padded_height=544, width=14,
        base_dir=GIROS_DIR, root_dir=GIROS_ROOT_DIR,
        make_dataset=GIROSTensorDataset,
    ),
    "CAEZ": dict(
        channels=1, height=273, padded_height=288, width=10,
        base_dir=CAEZ_DIR, root_dir=CAEZ_ROOT_DIR,
        make_dataset=CAEZGridTensorDataset,
    ),
    "FRAUNHOFER": dict(
        channels=1, height=49, padded_height=64, width=6,
        base_dir=FRAUNHOFER_DIR, root_dir=FRAUNHOFER_ROOT_DIR,
        make_dataset=FraunhoferTensorDataset,
    ),
}


def get_spec(dataset_name):
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' no soportado. Opciones: {list(DATASETS)}")
    return DATASETS[dataset_name]


def models_dir(dataset_name, quantizer):
    """<dataset>/vqgan-models-{q}"""
    return os.path.join(get_spec(dataset_name)["base_dir"], VQGAN_MODELS_SUBDIR.format(q=quantizer))


def recon_dir(dataset_name, quantizer):
    """<dataset>/vqgand_{q}"""
    return os.path.join(get_spec(dataset_name)["base_dir"], VQGAN_RECON_SUBDIR.format(q=quantizer))


def fit_to_spec(x, spec):
    """
    (B, C, H, W) -> (B, C, padded_height, width):
      1. Recorta a las filas reales (quita el padding que traiga el dataset del VQ-VAE).
      2. Columnas: si faltan, replica los bordes (como en FAWN); si sobran, recorta.
      3. Rellena con ceros al final del eje vertical hasta padded_height.
    """
    x = x[:, :, :spec["height"], :]

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
    Varianza de los valores del train, sin contar las filas de padding.
    Como en el VQ-VAE: hasta 20.000 grids elegidos con semilla 0
    (FAWN, GIROS y Fraunhofer tienen menos, así que se usan todos).
    """
    n = min(max_grids, len(train_data))
    idx = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(0))[:n].tolist()
    loader = DataLoader(torch.utils.data.Subset(train_data, idx), batch_size=256,
                        num_workers=num_workers, shuffle=False,
                        collate_fn=make_collate_fn(spec))
    total_sum, total_sq, total_n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x[:, :, :spec["height"], :].double()
            total_sum += x.sum().item()
            total_sq += (x ** 2).sum().item()
            total_n += x.numel()
    mean = total_sum / total_n
    return float(total_sq / total_n - mean ** 2), n


def load_data(args, compute_variance=True):
    """
    Devuelve: train_loader, val_loader, test_data, x_train_var
    Los grids salen como (B, C, padded_height, width) y los nombres como lista.
    """
    spec = get_spec(args.dataset)
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


def plot_comparison(real_imgs, recon_imgs, height, save_path=None):
    """Original frente a reconstrucción de la primera muestra del lote (sin padding)."""
    real = to_image(real_imgs[0, :, :height, :].detach().float().cpu())
    recon = to_image(recon_imgs[0, :, :height, :].detach().float().cpu())

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
