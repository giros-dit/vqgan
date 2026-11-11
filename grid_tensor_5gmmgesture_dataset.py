import argparse
import glob
import os
import torch
from torch.utils.data import Dataset

# 5GmmGesture (Bhat et al., mmHSense, IEEE Access 2026): potencia por par de haces (PPBP).
# Cada fichero PPBP-u{u}-e{e}-g{g}-t{t}.pt es (1, 50, 56) en float8:
#   50 filas = haces de transmisión, 56 columnas = haces de recepción
MMGESTURE_CHANNELS = 1
MMGESTURE_TX_BEAMS = 50
MMGESTURE_RX_BEAMS = 56
MMGESTURE_PATTERN = "PPBP-u*-e*-g*-t*.pt"


def list_5gmmgesture_files(root_dir, verbose=False):
    """Lista ordenada de los ficheros (misma lista en el entrenamiento y en el compressor)."""
    files = sorted(glob.glob(os.path.join(root_dir, MMGESTURE_PATTERN)))
    if len(files) == 0:
        raise RuntimeError(f"No se encontraron ficheros '{MMGESTURE_PATTERN}' en: {root_dir}")
    if verbose:
        print(f"Found {len(files)} ficheros PPBP")
    return files


def load_5gmmgesture(path, dtype=torch.float32):
    """Carga una grid sin cambiar su forma. Falla si no es un tensor."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"{os.path.basename(path)} no es un tensor: {type(obj).__name__}")
    return obj.to(dtype)


class MMGestureTensorDataset(Dataset):
    """Devuelve (1, 50, 56) en float32 y el nombre del fichero. El padding lo hace fit_to_spec."""

    def __init__(self, root_dir, data_type=torch.float32):
        self.root_dir = root_dir
        self.data_type = data_type
        self.tensor_files = list_5gmmgesture_files(root_dir, verbose=True)

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        f = self.tensor_files[idx]
        x = load_5gmmgesture(f, self.data_type)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        expected = (MMGESTURE_CHANNELS, MMGESTURE_TX_BEAMS, MMGESTURE_RX_BEAMS)
        if tuple(x.shape) != expected:
            raise ValueError(f"{os.path.basename(f)}: se esperaba {expected}, llegó {tuple(x.shape)}")
        label = os.path.splitext(os.path.basename(f))[0]
        return x, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MMGestureTensorDataset")
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    dataset = MMGestureTensorDataset(args.dataset_path)
    raw = torch.load(dataset.tensor_files[0], weights_only=False)
    print(f"Fichero en disco: forma {tuple(raw.shape)}, dtype {raw.dtype}")
    sample, label = dataset[torch.randint(0, len(dataset), (1,)).item()]
    print(f"Muestra: {tuple(sample.shape)}, {sample.dtype}, nombre: {label}")
    assert sample.shape == (MMGESTURE_CHANNELS, MMGESTURE_TX_BEAMS, MMGESTURE_RX_BEAMS)
    print("[OK] Forma (1, 50, 56) correcta.")