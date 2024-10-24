"""
Dataset file structure:

preprocessed_commoncatalog-cc-by/
    160x384/
        part-00000-tid-2529612224269674922-886aaa48-18bf-4e71-a1cf-5d3052f060f5-384892-1-c000.parquet
        ...
        part-00000-tid-6536873935347636966-7b25b5a4-6f41-419c-83d4-c8e00c6b11b2-454189-1-c000.parquet
    176x352
    192x320
    ...
    352x176

Custom dataloader (CommonCatalogDataLoader):

Has all the resolution folders as seperate dataloaders internally, shuffles the internal dataloaders, and randomly samples from one of the dataloaders for a batch.
"""

import os
import pyarrow.parquet as pq
import random
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List, Optional, Tuple, Dict, Union
from datasets import Dataset as HFDataset
from datasets import load_dataset
from huggingface_hub import HfFileSystem
from config import DS_DIR_BASE, DATASET_NAME, USERNAME
import numpy as np
import lightning as L
from torch.utils.data.distributed import DistributedSampler

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


class CommonCatalogDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule that handles multiple resolution datasets.
    """
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        transform=None,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
    def setup(self, stage: Optional[str] = None):
        """
        Load datasets and set up sampling weights.
        """
        self.datasets = get_datasets()
        
        # Calculate weights based on dataset lengths
        lengths = [len(ds) for ds in self.datasets]
        total_samples = sum(lengths)
        self.sampling_weights = [length / total_samples for length in lengths]
        
        # Calculate batches per dataset
        self.batches_per_dataset = [length // self.batch_size for length in lengths]
        
        # Create samplers for distributed training
        self.samplers = None
        if self.trainer and self.trainer.use_distributed_sampler:
            self.samplers = [
                DistributedSampler(
                    dataset,
                    num_replicas=self.trainer.world_size,
                    rank=self.trainer.global_rank,
                    shuffle=True
                )
                for dataset in self.datasets
            ]
    
    def _create_dataloader(self, dataset_idx: int) -> DataLoader:
        """
        Create a dataloader for a specific resolution dataset.
        """
        sampler = self.samplers[dataset_idx] if self.samplers else None
        
        return DataLoader(
            self.datasets[dataset_idx],
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True  # Simplify by dropping partial batches
        )
    
    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """
        Returns a list of dataloaders, one per resolution.
        """
        if self.samplers:
            for sampler in self.samplers:
                sampler.set_epoch(self.trainer.current_epoch)
        
        return [self._create_dataloader(i) for i in range(len(self.datasets))]

class ResolutionSamplingCallback(pl.Callback):
    """
    Lightning callback to handle resolution sampling during training.
    """
    def __init__(self):
        super().__init__()
        self.current_loaders = None
        self.batch_counts = None
        
    def setup_dataloaders(self, dataloaders):
        """
        Initialize with the list of dataloaders.
        """
        self.current_loaders = [iter(loader) for loader in dataloaders]
        self.batch_counts = [0] * len(dataloaders)
        
    def on_train_epoch_start(self, trainer, pl_module):
        """
        Reset dataloaders at the start of each epoch.
        """
        dataloaders = trainer.datamodule.train_dataloader()
        self.setup_dataloaders(dataloaders)
        
    def get_next_batch(self, trainer, pl_module):
        """
        Get next batch using weighted sampling.
        """
        weights = trainer.datamodule.sampling_weights
        
        # Adjust weights for available datasets
        available_indices = [i for i, loader in enumerate(self.current_loaders)
                           if self.batch_counts[i] < trainer.datamodule.batches_per_dataset[i]]
        
        if not available_indices:
            return None
            
        # Normalize weights for available datasets
        valid_weights = [weights[i] for i in available_indices]
        total = sum(valid_weights)
        valid_weights = [w / total for w in valid_weights]
        
        # Sample dataset
        dataset_idx = np.random.choice(available_indices, p=valid_weights)
        
        try:
            batch = next(self.current_loaders[dataset_idx])
            self.batch_counts[dataset_idx] += 1
            return batch
        except StopIteration:
            # This shouldn't happen due to our available_indices check
            return self.get_next_batch(trainer, pl_module)