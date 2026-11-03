import os

# Ruta base del almacenamiento de datos, elegida automáticamente:
#   1. Variable de entorno DISCO6G_DATA, si está definida (para forzar una ruta concreta)
#   2. /data, si existe (servidor con GPU)
#   3. /home/alfonso.inigo/disco6g-data (local)
_GPU_DATA_DIR = "/data"
_LOCAL_DATA_DIR = "/home/alfonso.inigo/disco6g-data"

if os.environ.get("DISCO6G_DATA"):
    DATA_BASE_DIR = os.environ["DISCO6G_DATA"]
elif os.path.isdir(os.path.join(_GPU_DATA_DIR, "datasets")):
    DATA_BASE_DIR = _GPU_DATA_DIR
else:
    DATA_BASE_DIR = _LOCAL_DATA_DIR

print(f"--> DATA_BASE_DIR = {DATA_BASE_DIR}")

_D = os.path.join(DATA_BASE_DIR, "datasets")

# Carpeta base de cada dataset (ahí van también los modelos y reconstrucciones)
FAWN_DIR = os.path.join(_D, "medidas-fawn/3gpp")
GIROS_DIR = os.path.join(_D, "GIROS/dataset_12-03")
CAEZ_DIR = os.path.join(_D, "caez/caez_5g_outdoor")
FRAUNHOFER_DIR = os.path.join(_D, "fraunhofer5G")

# Originales (los mismos que usa el VQ-VAE)
FAWN_ROOT_DIR = os.path.join(FAWN_DIR, "tensored/torch.float8_e4m3fn")
GIROS_ROOT_DIR = os.path.join(GIROS_DIR, "tensored_grid/torch.float32")
CAEZ_ROOT_DIR = os.path.join(CAEZ_DIR, "tensored/torch.float8_e4m3fn")
FRAUNHOFER_ROOT_DIR = os.path.join(FRAUNHOFER_DIR, "preprocessed_test_tensored/torch.float8_e4m3fn")

# Modelos y reconstrucciones del VQGAN, separados por cuantizador.
# {q} se sustituye por "ema" u "original":
#   <dataset>/vqgan-models-{q}/nembeddings{K}/model.pth
#   <dataset>/vqgand_{q}/torch.float8_e4m3fn_{K}/
VQGAN_MODELS_SUBDIR = "vqgan-models-{q}"
VQGAN_RECON_SUBDIR = "vqgand_{q}"