import torch
from torch.utils.data import IterableDataset
from collections import defaultdict
import numpy as np

class ShapeBatchingDataset(IterableDataset):
    def __init__(self, hf_dataset, batch_size, shuffle=True, seed=42, buffer_multiplier=20):
        self.dataset = hf_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.buffer_multiplier = buffer_multiplier

    def __iter__(self):
        if self.shuffle:
            self.dataset = self.dataset.shuffle(seed=self.seed, buffer_size=self.batch_size*self.buffer_multiplier)
        
        shape_batches = defaultdict(list)
        for sample in self.dataset:
            # Get the shape as a tuple to use as a key
            shape_key = tuple(sample['vae_latent_shape'])
            shape_batches[shape_key].append(sample) 

            # If enough samples are accumulated for this shape, yield a batch
            if len(shape_batches[shape_key]) == self.batch_size:
                batch = self.prepare_batch(shape_batches[shape_key])
                yield batch
                shape_batches[shape_key] = []  # Reset the buffer for this shape

        # After iterating over the dataset, yield any remaining partial batches
        for remaining_samples in shape_batches.values():
            if remaining_samples:
                batch = self.prepare_batch(remaining_samples)
                yield batch

    def prepare_batch(self, samples):
        # Convert lists of samples into tensors
        vae_latent_shape = tuple(samples[0]['vae_latent_shape'])

        batch = {
            'caption': [s['caption'] for s in samples],
            'vae_latent': torch.tensor(np.stack([np.frombuffer(s['vae_latent'], dtype=np.float32).copy() for s in samples])).reshape(-1, *vae_latent_shape),
            'vae_latent_shape': vae_latent_shape,
            'text_embedding': torch.tensor(np.stack([np.frombuffer(s['text_embedding'], dtype=np.float16).copy() for s in samples])),
        }
        return batch
    
    def __len__(self):
        return len(self.dataset) // self.batch_size
    
class Coco30kShapeBatchingDataset(IterableDataset):
    def __init__(self, hf_dataset, batch_size, shuffle, seed):
        self.dataset = hf_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.dataset = self.dataset.shuffle(seed=self.seed)
        
        shape_batches = defaultdict(list)
        for sample in self.dataset:
            # Get the shape as a tuple to use as a key
            shape_key = tuple(sample['vae_latent_shape'][1:])
            shape_batches[shape_key].append(sample) 

            # If enough samples are accumulated for this shape, yield a batch
            if len(shape_batches[shape_key]) == self.batch_size:
                batch = self.prepare_batch(shape_batches[shape_key])
                yield batch
                shape_batches[shape_key] = []  # Reset the buffer for this shape

        # After iterating over the dataset, yield any remaining partial batches
        for remaining_samples in shape_batches.values():
            if remaining_samples:
                batch = self.prepare_batch(remaining_samples)
                yield batch

    def prepare_batch(self, samples):
        # Convert lists of samples into tensors
        vae_latent_shape = tuple(samples[0]['vae_latent_shape'][1:])

        batch = {
            'caption': [s['caption'] for s in samples],
            'vae_latent': torch.tensor([s['vae_latent'] for s in samples], dtype=torch.float16).reshape(-1, *vae_latent_shape),
            'vae_latent_shape': vae_latent_shape,
            'text_embedding': torch.tensor([s['text_embedding'] for s in samples], dtype=torch.float16),
        }
        return batch
    
    def __len__(self):
        return len(self.dataset) // self.batch_size