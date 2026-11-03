import argparse
import glob
import os
import re
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Fraunhofer 5G (Stahlke et al., 2023): módulo de la CIR de 6 estaciones base.
# Cada fichero H-<n>.pt es (1, 49, 6) o (49, 6) en float8:
#   49 filas = instantes de la CIR (retardo), 6 columnas = estaciones base
FRAUNHOFER_CHANNELS = 1
FRAUNHOFER_TAPS = 49
FRAUNHOFER_ANCHORS = 6
# El encoder reduce x4 en vertical: 49 no es divisible entre 4 -> 52 = 13 * 4
FRAUNHOFER_PADDED_TAPS = 52


def _indice(path):
    """Número del fichero H-<n>.pt, para ordenar 0, 1, 2... y no 0, 1, 10, 100..."""
    m = re.search(r"H-(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def list_fraunhofer_files(root_dir, verbose=False):
    """Lista ordenada de los ficheros H-<n>.pt (misma lista en main.py y compressor.py)."""
    files = glob.glob(os.path.join(root_dir, "H-*.pt"))
    if len(files) == 0:
        raise RuntimeError(f"No se encontraron ficheros H-*.pt en: {root_dir}")
    files = sorted(files, key=_indice)
    if verbose:
        print(f"Found {len(files)} ficheros H-*.pt")
    return files


def load_fraunhofer(path, dtype=torch.float32):
    """Carga un fichero sin cambiar su forma. Falla si no es un tensor."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"{os.path.basename(path)} no es un tensor: {type(obj).__name__}")
    return obj.to(dtype)


def to_fraunhofer_shape(x):
    """Asegura forma (1, 49, 6)."""
    if x.ndim == 2:
        x = x.unsqueeze(0)
    expected = (FRAUNHOFER_CHANNELS, FRAUNHOFER_TAPS, FRAUNHOFER_ANCHORS)
    if tuple(x.shape) != expected:
        raise ValueError(f"Se esperaba {expected} o (49, 6), llegó {tuple(x.shape)}")
    return x


def pad_taps(x, padded_taps=FRAUNHOFER_PADDED_TAPS):
    """Filas de ceros al final del eje del tiempo hasta padded_taps."""
    pad_h = padded_taps - x.shape[-2]
    if pad_h > 0:
        x = F.pad(x, (0, 0, 0, pad_h), mode="constant", value=0.0)
    return x


class FraunhoferTensorDataset(Dataset):
    """Devuelve (1, 52, 6) en float32 (49 filas reales + 3 de ceros) y el nombre del fichero."""

    def __init__(self, root_dir, data_type=torch.float32):
        self.root_dir = root_dir
        self.data_type = data_type
        self.tensor_files = list_fraunhofer_files(root_dir, verbose=True)

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        f = self.tensor_files[idx]
        x = load_fraunhofer(f, self.data_type)
        x = to_fraunhofer_shape(x)
        x = pad_taps(x)
        label = os.path.splitext(os.path.basename(f))[0]
        return x, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FraunhoferTensorDataset")
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    dataset = FraunhoferTensorDataset(args.dataset_path)
    print(f"Dataset cargado: {len(dataset)} muestras")

    raw = torch.load(dataset.tensor_files[0], weights_only=False)
    print(f"Fichero en disco: forma {tuple(raw.shape)}, dtype {raw.dtype}")

    sample, label = dataset[torch.randint(0, len(dataset), (1,)).item()]
    print(f"Muestra tras padding: {tuple(sample.shape)}, {sample.dtype}, nombre: {label}")
    assert sample.shape == (1, FRAUNHOFER_PADDED_TAPS, FRAUNHOFER_ANCHORS)
    assert torch.all(sample[:, FRAUNHOFER_TAPS:, :] == 0)
    print("[OK] Forma (1, 52, 6) y padding correctos.")