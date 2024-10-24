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
from typing import Any, List, Optional, Tuple, Dict, Union
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

class CommonCatalogDataset(IterableDataset):
    """
    Dataset that handles multiple resolution datasets with proper sampling.
    Implements IterableDataset to handle sampling internally.
    """
    def __init__(
        self,
        batch_size: int,
        seed: Optional[int] = None,
        world_size: int = 1,
        rank: int = 0,
        shuffle: bool = True
    ):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        # Load datasets
        self.datasets = get_datasets()
        
        # Calculate dataset lengths and weights
        self.lengths = [len(ds) for ds in self.datasets]
        total_samples = sum(self.lengths)
        self.weights = [length / total_samples for length in self.lengths]
        
        # Calculate number of batches
        self.batches_per_dataset = [length // batch_size for length in self.lengths]
        self.total_batches = sum(self.batches_per_dataset)
        
        if self.world_size > 1:
            self.total_batches = self.total_batches // self.world_size
            
        # Create indices for each dataset
        self.dataset_indices = [list(range(length)) for length in self.lengths]
    
    def __len__(self) -> int:
        return self.total_batches
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch number for proper shuffling in distributed training."""
        self.epoch = epoch
        
    def _get_shuffled_indices(self) -> List[List[int]]:
        """Get shuffled indices for each dataset, properly seeded for distributed training."""
        if not self.shuffle:
            return self.dataset_indices
            
        # Create deterministic shuffle based on epoch and rank
        shuffled_indices = []
        for i, indices in enumerate(self.dataset_indices):
            rand = random.Random(hash((self.epoch, self.rank, i)))
            shuffled = indices.copy()
            rand.shuffle(shuffled)
            shuffled_indices.append(shuffled)
            
        return shuffled_indices
    
    def __iter__(self):
        # Set up shuffled indices
        shuffled_indices = self._get_shuffled_indices()
        current_indices = [0] * len(self.datasets)
        
        # Calculate how many samples each GPU should process
        samples_per_gpu = [length // self.world_size for length in self.lengths]
        start_idx = [self.rank * (length // self.world_size) for length in self.lengths]
        end_idx = [(self.rank + 1) * (length // self.world_size) for length in self.lengths]
        
        batches_yielded = 0
        
        while batches_yielded < self.total_batches:
            # Sample a dataset based on weights and available samples
            available_datasets = []
            available_weights = []
            
            for i, (start, end, current) in enumerate(zip(start_idx, end_idx, current_indices)):
                if current + self.batch_size <= end:
                    available_datasets.append(i)
                    available_weights.append(self.weights[i])
                    
            if not available_datasets:
                break
                
            # Normalize weights
            total_weight = sum(available_weights)
            available_weights = [w / total_weight for w in available_weights]
            
            # Sample dataset
            dataset_idx = np.random.choice(available_datasets, p=available_weights)
            
            # Get batch indices
            batch_start = current_indices[dataset_idx]
            batch_end = min(batch_start + self.batch_size, end_idx[dataset_idx])
            batch_indices = shuffled_indices[dataset_idx][batch_start:batch_end]
            
            # Update current index
            current_indices[dataset_idx] = batch_end
            
            # Yield batch
            if len(batch_indices) == self.batch_size:
                batch = [self.datasets[dataset_idx][idx] for idx in batch_indices]
                batches_yielded += 1
                yield self.collate_batch(batch)
    
    def collate_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples into a single batch dictionary."""
        return {
            key: torch.stack([sample[key] for sample in batch])
            for key in batch[0].keys()
        }

class CommonCatalogDataModule(L.LightningDataModule):
    """
    Lightning DataModule that uses CommonCatalogDataset.
    """
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
    
    def setup(self, stage: Optional[str] = None):
        """Load datasets for quick access to examples, etc."""
        self.datasets = get_datasets()
    
    def train_dataloader(self) -> DataLoader:
        world_size = 1
        rank = 0
        
        # Check if we're in a distributed setting
        if self.trainer:
            strategy = getattr(self.trainer, 'strategy', None)
            if strategy and not isinstance(strategy, L.strategies.SingleDeviceStrategy):
                world_size = self.trainer.world_size
                rank = self.trainer.global_rank
            
        dataset = CommonCatalogDataset(
            batch_size=self.batch_size,
            seed=self.seed,
            world_size=world_size,
            rank=rank,
            shuffle=True
        )
        
        # Create DataLoader with the custom dataset
        return DataLoader(
            dataset,
            batch_size=None,  # Batching is handled by the dataset
            num_workers=self.num_workers,
            pin_memory=True
        )