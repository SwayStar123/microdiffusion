import sys
sys.path.append('../dcae')
from dcae import DCAE

from datasets import load_dataset
import datasets
# from dataset.shapebatching_dataset import ShapeBatchingDataset
from time import time
from config import DATASET_NAME, USERNAME, DS_DIR_BASE, VAE_SCALING_FACTOR, MODELS_DIR_BASE
import torch
import numpy as np
import torchvision

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

datasets.config.HF_HUB_OFFLINE = 1

dataset = load_dataset(f"{USERNAME}/{DATASET_NAME}", split="train", cache_dir=f"{DS_DIR_BASE}/{DATASET_NAME}")
# dataset = ShapeBatchingDataset(dataset, 768, True, 0)
dc_ae = DCAE("dc-ae-f32c32-mix-1.0", device=DEVICE, dtype=DTYPE, cache_dir=f"{MODELS_DIR_BASE}/dc_ae").eval()

def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)

latent = next(iter(dataset))["latent"]
latent = torch.tensor(latent, device=DEVICE, dtype=DTYPE)

with torch.no_grad():
        recon = dc_ae.decode(latent.unsqueeze(0)).squeeze(0)

recon = denorm(recon).to(torch.float32)

torchvision.utils.save_image(
    recon,
    "recon.png",
    normalize=False,
)