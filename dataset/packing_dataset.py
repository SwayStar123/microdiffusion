import random
from torch.utils.data import IterableDataset
from datasets import load_dataset
from huggingface_hub import HfFileSystem
from config import USERNAME, DATASET_NAME, DS_DIR_BASE
import torch
import numpy as np

def get_datasets():
    fs = HfFileSystem()
    objs = fs.ls(f"datasets/{USERNAME}/{DATASET_NAME}", detail=False)
    folders = [obj for obj in objs if fs.isdir(obj)]

    datasets = []
    for folder in folders:
        folder_name = folder.split('/')[-1]
        ds = load_dataset(f"{USERNAME}/{DATASET_NAME}", data_dir=folder_name, split="train", cache_dir=f"{DS_DIR_BASE}/{DATASET_NAME}", num_proc=32)
        datasets.append(ds)

    return datasets

class CombinedDataset(IterableDataset):
    def __init__(self, datasets, seed, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        dataset, = self._rng.choices(self._datasets, weights=self._weights, k=1)

        batch = next(dataset)

        vae_latent_shape = tuple(batch['vae_latent_shape'][0])
        batch["vae_latent"] = torch.tensor(np.stack([np.frombuffer(s, dtype=np.float16).copy() for s in batch['vae_latent']])).reshape(-1, *vae_latent_shape),
        batch["text_embedding"] = torch.tensor(np.stack([np.frombuffer(s, dtype=np.float16).copy() for s in batch['text_embedding']])),

        return batch