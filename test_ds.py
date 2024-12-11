# import sys
# sys.path.append('../dcae')
# from dcae import DCAE
from datasets import load_dataset
import datasets
# from dataset.shapebatching_dataset import ShapeBatchingDataset
from time import time
from config import DATASET_NAME, USERNAME, DS_DIR_BASE, VAE_SCALING_FACTOR, MODELS_DIR_BASE
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader

from transformer.utils import apply_mask_to_tensor, random_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

datasets.config.HF_HUB_OFFLINE = 1

dataset = load_dataset(f"{USERNAME}/{DATASET_NAME}", split="train", cache_dir=f"{DS_DIR_BASE}/{DATASET_NAME}")
dataset = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

patch_size = (1,1)

def batch_to_tensors(batch):
    latents = batch["latent"]
    latents = torch.stack(
    [torch.stack([torch.stack(inner) for inner in outer]) for outer in latents]
    )
    latents = latents.permute(3, 0, 1, 2) # for some reason batch size is last so we need to permute it
    return latents

last = time()
for i, batch in enumerate(dataset):
    latents = batch_to_tensors(batch).to(DEVICE, dtype=DTYPE)
    # caption_embeddings = batch["text_embedding"].to(device)
    bs = latents.shape[0]

    # Null caption embeddings for this dataset
    caption_embeddings = torch.zeros((bs, 768), device=DEVICE, dtype=DTYPE)

    latents = latents * VAE_SCALING_FACTOR

    mask = random_mask(bs, latents.shape[-2], latents.shape[-1], patch_size, mask_ratio=0.75).to(DEVICE, dtype=DTYPE)

    nt = torch.randn((bs,)).to(DEVICE, dtype=DTYPE)
    t = torch.sigmoid(nt)
    
    texp = t.view([bs, *([1] * len(latents.shape[1:]))]).to(DEVICE, dtype=DTYPE)
    z1 = torch.randn_like(latents, device=DEVICE, dtype=DTYPE)
    zt = (1 - texp) * latents + texp * z1
    

    latents = apply_mask_to_tensor(latents, mask, patch_size)
    print(f"{i}: {time() - last}")
    last = time()
    pass



# dataset = ShapeBatchingDataset(dataset, 768, True, 0)
# dc_ae = DCAE("dc-ae-f32c32-mix-1.0", device=DEVICE, dtype=DTYPE, cache_dir=f"{MODELS_DIR_BASE}/dc_ae").eval()

# def denorm(x):
#     return (x * 0.5 + 0.5).clamp(0, 1)

# latent = next(iter(dataset))["latent"]
# latent = torch.tensor(latent, device=DEVICE, dtype=DTYPE)

# with torch.no_grad():
#         recon = dc_ae.decode(latent.unsqueeze(0)).squeeze(0)

# recon = denorm(recon).to(torch.float32)

# torchvision.utils.save_image(
#     recon,
#     "recon.png",
#     normalize=False,
# )