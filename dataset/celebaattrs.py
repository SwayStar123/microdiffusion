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
        if isinstance(idx, slice):
            # Generate a list of indices from the slice
            indices = range(*idx.indices(len(self)))
            latents = []
            prompts = []
            for i in indices:
                latent_path = os.path.join(self.dataset_path, f"latents/{i:08d}.pt")
                latents.append(torch.load(latent_path))
                prompts.append(self.metadata[str(i)])
            # Stack latents into a single tensor and return with prompts
            return torch.stack(latents), prompts
        else:
            latent_path = os.path.join(self.dataset_path, f"latents/{idx:08d}.pt")
            latents = torch.load(latent_path)
            prompt = self.metadata[str(idx)]
            return latents, prompt