from collections import defaultdict
import os
import pyarrow.parquet as pq
import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler, SequentialSampler, RandomSampler
from typing import Any, List, Optional, Tuple, Dict, Union
from datasets import Dataset as HFDataset
from datasets import load_dataset
from huggingface_hub import HfFileSystem, hf_hub_download
from config import DS_DIR_BASE, USERNAME
import numpy as np
import lightning as L
from torch.utils.data.distributed import DistributedSampler
from lightning.pytorch.strategies import SingleDeviceStrategy
import time
import glob
from concurrent.futures import ThreadPoolExecutor

DATASET_NAME = "preprocessed_recap-coco30k-moondream"

def download_with_retry(repo_id, filename, local_dir, max_retries=500, retry_delay=60):
    for attempt in range(max_retries):
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=local_dir)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Download failed. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries}). Exception: {e}")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached. Skipping file: {filename}")
                return None
            
def download_file(folder_name, file, downloaded_files):
    file = file.split('/')[-1]
    folder_file = f"{folder_name}/{file}"
    if folder_file not in downloaded_files:
        download_with_retry(repo_id=f"{USERNAME}/{DATASET_NAME}", filename=folder_file, local_dir=f"{DS_DIR_BASE}/{DATASET_NAME}")

def get_datasets(num_proc=16):
    fs = HfFileSystem()
    objs = fs.ls(f"datasets/{USERNAME}/{DATASET_NAME}", detail=False)
    folders = [obj for obj in objs if fs.isdir(obj)]

    # Get all .parquet files already in {DS_DIR_BASE}/{DATASET_NAME}/**/*.parquet
    downloaded_files = glob.glob(f"{DS_DIR_BASE}/{DATASET_NAME}/**/*.parquet", recursive=True)
    # Strip down to just last folder and filename
    downloaded_files = [os.path.join(*path.split(os.sep)[-2:]) for path in downloaded_files]

    datasets = []
    for folder in folders:
        folder_name = folder.split('/')[-1]
        files = fs.ls(folder, detail=False)

        with ThreadPoolExecutor(max_workers=num_proc) as executor:
            executor.map(lambda file: download_file(folder_name, file, downloaded_files), files)

        ds = load_dataset(f"{DS_DIR_BASE}/{DATASET_NAME}/{folder_name}", split="train")
        datasets.append(ds)

    return datasets

from datasets import concatenate_datasets

class CustomDataset(Dataset):
    def __init__(self, datasets):
        """
        Args:
            datasets (list): List of Hugging Face datasets.
        """
        # Concatenate the datasets
        self.dataset = concatenate_datasets(datasets)
        # Store the sizes (vae_latent_shape) for each example
        self.sizes = []
        for idx in range(len(self.dataset)):
            vae_latent_shape = tuple(self.dataset[idx]['vae_latent_shape'])
            self.sizes.append(vae_latent_shape)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        # Convert vae_latent to tensor and reshape
        vae_latent = torch.tensor(example['vae_latent'])
        vae_latent_shape = tuple(example['vae_latent_shape'])
        vae_latent = vae_latent.view(*vae_latent_shape)
        # Convert text_embedding to tensor
        text_embedding = torch.tensor(example['text_embedding'])
        return vae_latent, text_embedding

class CustomBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, dataset, shuffle=True):
        """
        Args:
            sampler (Sampler): Base sampler (e.g., RandomSampler).
            batch_size (int): Number of samples per batch.
            drop_last (bool): Whether to drop the last incomplete batch.
            dataset (Dataset): The dataset to sample from.
            shuffle (bool): Whether to shuffle the data.
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset = dataset
        self.shuffle = shuffle
        self.sizes = self.dataset.sizes
        self.batches = self.create_batches()
    
    def create_batches(self):
        size_to_indices = defaultdict(list)
        # Collect indices from the base sampler
        indices = list(self.sampler)
        # Group indices by vae_latent_shape
        for idx in indices:
            size = self.sizes[idx]
            size_to_indices[size].append(idx)
        batches = []
        for size, idxs in size_to_indices.items():
            if self.shuffle:
                random.shuffle(idxs)
            # Create batches from indices of the same size
            batch_indices = [idxs[i:i + self.batch_size] 
                             for i in range(0, len(idxs), self.batch_size)]
            if self.drop_last:
                batch_indices = [batch for batch in batch_indices if len(batch) == self.batch_size]
            batches.extend(batch_indices)
        if self.shuffle:
            random.shuffle(batches)  # Shuffle the list of batches
        return batches
    
    def __iter__(self):
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)