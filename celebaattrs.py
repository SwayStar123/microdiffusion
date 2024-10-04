import torch
from torch.utils.data import Dataset
import os
import json

dataset_relative_path = "../../datasets/CelebA-attrs-latents"

class CelebAAttrsDataset(Dataset):
    def __init__(self, set):
        self.dataset_path = dataset_relative_path + "/" + set
        self.metadata = json.load(open(self.dataset_path + "/metadata.json"))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        latent_path = os.path.join(self.dataset_path, f"latents/{idx:08d}.pt")
        latents = torch.load(latent_path)
        prompt = self.metadata[str(idx)]
        return latents, prompt