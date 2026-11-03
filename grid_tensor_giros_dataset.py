import argparse
import glob
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# GIROS (dataset_12-03): 2 canales (I/Q) x 540 subportadoras x 14 símbolos OFDM
GIROS_CHANNELS = 2
GIROS_SUBCARRIERS = 540   # divisible entre 4 -> latente 135 x 14, sin padding
GIROS_SYMBOLS = 14


def list_giros_files(root_dir, verbose=False):
    """Lista ordenada de los .pt (misma lista en main.py y compressor.py)."""
    files = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
    if len(files) == 0:
        raise RuntimeError(f"No se encontraron ficheros .pt en: {root_dir}")
    if verbose:
        print(f"Found {len(files)} ficheros .pt")
    return files


def load_giros(path, dtype=torch.float32):
    """Carga un grid sin cambiar su forma. Falla si no es un tensor."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"{os.path.basename(path)} no es un tensor: {type(obj).__name__}")
    return obj.to(dtype)


def to_giros_shape(x):
    """Asegura (2, 540, 14): recorta si sobra y rellena con ceros si falta."""
    if x.ndim != 3 or x.shape[0] != GIROS_CHANNELS:
        raise ValueError(f"Se esperaba (2, H, W), llegó {tuple(x.shape)}")
    x = x[:, :GIROS_SUBCARRIERS, :GIROS_SYMBOLS]
    pad_h = GIROS_SUBCARRIERS - x.shape[1]
    pad_w = GIROS_SYMBOLS - x.shape[2]
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return x


class GIROSTensorDataset(Dataset):
    """Devuelve (2, 540, 14) en float32 y el nombre del fichero (sin etiquetas de presencia)."""

    def __init__(self, root_dir, data_type=torch.float32):
        self.root_dir = root_dir
        self.data_type = data_type
        self.tensor_files = list_giros_files(root_dir, verbose=True)

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        f = self.tensor_files[idx]
        x = to_giros_shape(load_giros(f, self.data_type))
        label = os.path.splitext(os.path.basename(f))[0]
        return x, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GIROSTensorDataset")
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    dataset = GIROSTensorDataset(args.dataset_path)
    raw = torch.load(dataset.tensor_files[0], weights_only=False)
    print(f"Fichero en disco: forma {tuple(raw.shape)}, dtype {raw.dtype}")
    sample, label = dataset[torch.randint(0, len(dataset), (1,)).item()]
    print(f"Muestra: {tuple(sample.shape)}, {sample.dtype}, nombre: {label}")
    assert sample.shape == (GIROS_CHANNELS, GIROS_SUBCARRIERS, GIROS_SYMBOLS)
    print("[OK] Forma (2, 540, 14) correcta.")