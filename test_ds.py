from datasets import load_dataset
from dataset.shapebatching_dataset import ShapeBatchingDataset
from time import time
from config import DATASET_NAME, USERNAME, DS_DIR_BASE, VAE_SCALING_FACTOR

dataset = load_dataset(f"{USERNAME}/{DATASET_NAME}", split="train", streaming=True)
dataset = ShapeBatchingDataset(dataset, 768, True, 0)

for batch in dataset:
    latents = batch["vae_latent"]
    # Get std dev, mean, min, max
    print(latents.std(), latents.mean(), latents.min(), latents.max())

    latents = latents * VAE_SCALING_FACTOR
    print(latents.std(), latents.mean(), latents.min(), latents.max())

    embedding = batch["text_embedding"]
    print(embedding.std(), embedding.mean(), embedding.min(), embedding.max())