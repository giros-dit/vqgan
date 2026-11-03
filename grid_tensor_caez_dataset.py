import argparse
import glob
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Dimensiones de CAEZ 5G outdoor: 1 canal x 273 subportadoras x 10 símbolos OFDM
CAEZ_CHANNELS = 1
CAEZ_SUBCARRIERS = 273
CAEZ_SYMBOLS = 10
# 273 no es divisible entre 4 (el encoder reduce x4 en frecuencia) -> 276 = 69 * 4
CAEZ_PADDED_SUBCARRIERS = 276


class CAEZGridTensorDataset(Dataset):
    """
    Dataset para CAEZ 5G outdoor.
    Cada fichero .pt es una rejilla de 1 canal x 273 subportadoras x 10 símbolos
    (se acepta también sin dimensión de canal: 273 x 10).

    Devuelve la rejilla en float32 con padding de ceros en frecuencia hasta 276
    subportadoras (3 filas al final), para que el encoder (x4) dé un latente 69 x 10.
    """

    def __init__(self, root_dir, data_type=torch.float32,
                 n_subcarriers=CAEZ_SUBCARRIERS,
                 n_symbols=CAEZ_SYMBOLS,
                 padded_subcarriers=CAEZ_PADDED_SUBCARRIERS):
        self.root_dir = root_dir
        self.data_type = data_type
        self.n_subcarriers = n_subcarriers
        self.n_symbols = n_symbols
        self.padded_subcarriers = padded_subcarriers

        self.tensor_files = list_caez_grid_files(root_dir, verbose=True)

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        f = self.tensor_files[idx]

        # float8 no admite F.pad ni la mayoría de operaciones -> pasar a float32 primero
        grid = load_grid(f, self.data_type if self.data_type is not None else torch.float32)

        grid = to_caez_shape(grid, self.n_subcarriers, self.n_symbols)
        grid = pad_subcarriers(grid, self.padded_subcarriers)

        # Etiqueta: nombre del fichero sin extensión (útil para trazar muestras)
        label = os.path.splitext(os.path.basename(f))[0]
        return grid, label


# En la carpeta conviven dos tipos de fichero:
#   H-t_<t>-B_<b>-A_<a>.pt -> grid de canal, tensor float8 (1, 273, 10)   <- los que queremos
#   H-t_<t>.pt             -> array NumPy float64 (10, 2), NO es un grid   <- se descartan
CAEZ_GRID_PATTERN = 'H-t_*-B_*-A_*.pt'


def list_caez_grid_files(root_dir, verbose=False):
    """Lista ordenada solo de los ficheros de grid de CAEZ (excluye los H-t_<t>.pt)."""
    all_files = glob.glob(os.path.join(root_dir, '*.pt'))
    grid_files = sorted(glob.glob(os.path.join(root_dir, CAEZ_GRID_PATTERN)))
    if len(grid_files) == 0:
        raise RuntimeError(f"No se encontraron grids '{CAEZ_GRID_PATTERN}' en: {root_dir}")
    if verbose:
        print(f"Found {len(all_files)} .pt files | {len(grid_files)} grids "
              f"({len(all_files) - len(grid_files)} descartados por no ser grids)")
    return grid_files


def load_grid(path, dtype=torch.float32):
    """Carga un grid de CAEZ. Falla si el fichero no contiene un tensor de PyTorch."""
    obj = torch.load(path, weights_only=False)
    if not isinstance(obj, torch.Tensor):
        shape = getattr(obj, 'shape', '-')
        raise TypeError(
            f"{os.path.basename(path)} no es un grid: contiene {type(obj).__name__} "
            f"con forma {shape}. Revisa el filtro CAEZ_GRID_PATTERN."
        )
    return obj.to(dtype)


def to_caez_shape(grid, n_subcarriers=CAEZ_SUBCARRIERS, n_symbols=CAEZ_SYMBOLS):
    """Asegura forma (1, n_subcarriers, n_symbols)."""
    if grid.ndim == 2:
        grid = grid.unsqueeze(0)
    if grid.ndim != 3:
        raise ValueError(f"Se esperaba (C, H, W) o (H, W), llegó {tuple(grid.shape)}")
    if grid.shape[0] != CAEZ_CHANNELS:
        raise ValueError(f"Se esperaba {CAEZ_CHANNELS} canal, llegó {tuple(grid.shape)}")

    if grid.shape[1] < n_subcarriers:
        raise ValueError(f"Se esperaban {n_subcarriers} subportadoras, llegó {tuple(grid.shape)}")

    grid = grid[:, :n_subcarriers, :n_symbols]

    # Si algún grid viene con menos símbolos, se replican los bordes (como en FAWN)
    w = grid.shape[-1]
    if w < n_symbols:
        diff = n_symbols - w
        grid = F.pad(grid.unsqueeze(0), (diff // 2, diff - diff // 2, 0, 0),
                     mode="replicate").squeeze(0)
    return grid


def pad_subcarriers(grid, padded_subcarriers=CAEZ_PADDED_SUBCARRIERS):
    """Añade filas de ceros al final del eje de subportadoras hasta padded_subcarriers."""
    pad_h = padded_subcarriers - grid.shape[-2]
    if pad_h > 0:
        grid = F.pad(grid, (0, 0, 0, pad_h), mode="constant", value=0.0)
    return grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CAEZGridTensorDataset")
    parser.add_argument('--dataset_path', type=str, required=True, help='Ruta al directorio de tensores .pt')
    args = parser.parse_args()

    dataset = CAEZGridTensorDataset(root_dir=args.dataset_path)
    print(f"Dataset cargado con éxito: {len(dataset)} rejillas.")

    raw = torch.load(dataset.tensor_files[0], weights_only=False)
    print(f"Fichero en disco: forma {tuple(raw.shape)}, dtype {raw.dtype}")

    random_idx = torch.randint(0, len(dataset), (1,)).item()
    sample, label = dataset[random_idx]
    print(f"Muestra tras padding: {tuple(sample.shape)}, dtype {sample.dtype}, nombre: {label}")
    assert sample.shape == (1, CAEZ_PADDED_SUBCARRIERS, CAEZ_SYMBOLS)
    assert torch.all(sample[:, CAEZ_SUBCARRIERS:, :] == 0), "Las filas de padding deben ser ceros"
    print("[OK] Forma (1, 276, 10) y padding correctos.")