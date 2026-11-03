import argparse
import glob
import os
import torch
from torch.utils.data import Dataset

coord_classes = [
    '1_1', '1_4', '1_5', '1_6', '1_7', '1_8',
    '2_1', '2_2', '2_3', '2_4', '2_5', '2_6', '2_7',
    '3_1', '3_2', '3_6', '3_7', '3_8',
    '4_1', '4_2', '4_7', '4_8',
    '5_1', '5_2', '5_3', '5_4', '5_5', '5_6', '5_7', '5_8', '5_9',
    '6_1', '6_2', '6_3', '6_4', '6_5', '6_6', '6_7', '6_8', '6_9',
    '7_1', '7_2', '7_3', '7_4', '7_5', '7_6', '7_7', '7_8', '7_9',
    '8_4', '8_5', '8_6', '8_7', '8_8', '8_9'
]


class FAWNGridSymbolTensorDataset(Dataset):
    """Custom Dataset para cargar rejillas completas (Caso A: 2 canales, 360 subportadoras, 6 símbolos)."""
    
    def __init__(self, root_dir, data_type=torch.float32):
        """
        Args:
            root_dir: Ruta al directorio con los archivos .pt
            data_type: Tipo de dato de salida (por defecto torch.float32)
        """
        self.root_dir = root_dir
        self.data_type = data_type
        self.classes = coord_classes
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.tensor_files = sorted(glob.glob(os.path.join(root_dir, '*.pt')))
        if len(self.tensor_files) == 0:
            raise RuntimeError(f"No se encontraron archivos .pt en: {root_dir}")
        
        print(f"Found {len(self.tensor_files)} tensor files in total")

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        f = self.tensor_files[idx]
        loaded_grid = torch.load(f, weights_only=False)

        # Extraer rejilla completa: 2 canales x 360 subportadoras x 6 símbolos
        # Forma resultante por muestra: (2, 360, 6)
        grid = loaded_grid[:, :360, :6]
        
        if self.data_type is not None:
            grid = grid.to(self.data_type)
        else:
            grid = grid.float()

        # Extraer coordenadas del nombre del archivo, e.g. "...(5_3)..." -> "5_3"
        label = f.split('/')[-1].split('(')[-1].split(')')[0]

        return grid, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FAWNGridSymbolTensorDataset")
    parser.add_argument('--dataset_path', type=str, required=True, help='Ruta al directorio de tensores .pt')
    parser.add_argument('--data_type', type=str, default='float32', help='Tipo de dato (float32, etc.)')
    args = parser.parse_args()

    data_type = getattr(torch, args.data_type)
    dataset = FAWNGridSymbolTensorDataset(
        root_dir=args.dataset_path,
        data_type=data_type
    )
    
    print(f"Dataset cargado con éxito: {len(dataset)} rejillas completas.")
    random_idx = torch.randint(0, len(dataset), (1,)).item()
    sample, label = dataset[random_idx]
    print(f"Forma de la muestra: {sample.shape}, Tipo: {sample.dtype}, Etiqueta: {label}")