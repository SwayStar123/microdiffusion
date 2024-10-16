import torch.nn as nn
from .embed import PatchEmbed, get_2d_sincos_pos_embed
from .utils import random_mask, remove_masked_patches, add_masked_patches, unpatchify, apply_mask_to_tensor
from .backbone import TransformerBackbone
from .moedit import TimestepEmbedder
import lightning as L
import torch
from torch.utils.data import DataLoader
import os
import json
import glob
import torch
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, DataLoader
from dataset.bucket_manager import BucketManager
from datasets import load_dataset
from config import VAE_SCALING_FACTOR, DS_DIR_BASE, METADATA_DATASET_NAME, DATASET_NAME

class PatchMixer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MicroDiT(nn.Module):
    """
    MicroDiT is a image diffusion transformer model.

    Args:
        in_channels (int): Number of input channels in the image data.
        patch_size (tuple of int): Size of the patches to be extracted from the image.
        embed_dim (int): Dimension of the embedding space.
        num_layers (int): Number of layers in the transformer backbone.
        num_heads (int): Number of attention heads in the multi-head attention mechanism.
        mlp_dim (int): Dimension of the multi-layer perceptron.
        class_label_dim (int): Dimension of the class labels.
        num_experts (int, optional): Number of experts in the transformer backbone. Default is 4.
        active_experts (int, optional): Number of active experts in the transformer backbone. Default is 2.
        dropout (float, optional): Dropout rate. Default is 0.1.
        patch_mixer_layers (int, optional): Number of layers in the patch mixer. Default is 2.
        embed_cat (bool, optional): Whether to concatenate embeddings. Default is False. If true, the timestep, class, and positional embeddings are concatenated rather than summed.

    Attributes:
        patch_size (tuple of int): Size of the patches to be extracted from the image.
        embed_dim (int): Dimension of the embedding space.
        patch_embed (PatchEmbed): Patch embedding layer.
        time_embed (TimestepEmbedder): Timestep embedding layer.
        class_embed (nn.Sequential): Class embedding layer.
        mha (nn.MultiheadAttention): Multi-head attention mechanism.
        mlp (nn.Sequential): Multi-layer perceptron for processing embeddings.
        pool_mlp (nn.Sequential): Pooling and multi-layer perceptron for (MHA + MLP).
        linear (nn.Linear): Linear layer after MHA+MLP.
        patch_mixer (PatchMixer): Patch mixer layer.
        backbone (TransformerBackbone): Transformer backbone model.
    """
    def __init__(self, in_channels, patch_size, embed_dim, num_layers, num_heads, mlp_dim, caption_embed_dim, timestep_caption_embed_dim=None, pos_embed_dim=None,
                 num_experts=4, active_experts=2, dropout=0.1, patch_mixer_layers=2, embed_cat=False):
        super().__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.embed_cat = embed_cat

        if self.embed_cat:
            self.pos_embed_dim = pos_embed_dim if pos_embed_dim is not None else embed_dim
            self.timestep_caption_embed_dim = timestep_caption_embed_dim if timestep_caption_embed_dim is not None else embed_dim
            self.total_embed_dim = self.timestep_caption_embed_dim + self.pos_embed_dim + embed_dim
        else:
            self.pos_embed_dim = embed_dim
            self.timestep_caption_embed_dim = embed_dim
            self.total_embed_dim = embed_dim
        
        # Image processing
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        
        # Timestep embedding
        self.time_embed = TimestepEmbedder(self.timestep_caption_embed_dim)
        
        # Caption embedding
        self.caption_embed = nn.Sequential(
            nn.Linear(caption_embed_dim, self.timestep_caption_embed_dim),
            nn.GELU(),
            nn.Linear(self.timestep_caption_embed_dim, self.timestep_caption_embed_dim)
        )

        # MHA for timestep and caption
        self.mha = nn.MultiheadAttention(self.timestep_caption_embed_dim, num_heads, batch_first=True)
        
        # MLP for timestep and caption
        self.mlp = nn.Sequential(
            nn.Linear(self.timestep_caption_embed_dim, self.timestep_caption_embed_dim),
            nn.GELU(),
            nn.Linear(self.timestep_caption_embed_dim, self.timestep_caption_embed_dim)
        )
        
        # Pool + MLP for (MHA + MLP)
        self.pool_mlp = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.timestep_caption_embed_dim, self.timestep_caption_embed_dim),
            nn.GELU(),
            nn.Linear(self.timestep_caption_embed_dim, self.timestep_caption_embed_dim)
        )
        
        # Linear layer after MHA+MLP
        self.linear = nn.Linear(self.timestep_caption_embed_dim, self.timestep_caption_embed_dim)
        
        # Patch-mixer
        self.patch_mixer = PatchMixer(self.total_embed_dim, num_heads, patch_mixer_layers)
        
        # Backbone transformer model
        self.backbone = TransformerBackbone(self.total_embed_dim, self.total_embed_dim, self.timestep_caption_embed_dim, num_layers, num_heads, mlp_dim, 
                                        num_experts, active_experts, dropout)
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(self.total_embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, patch_size[0] * patch_size[1] * in_channels)
        )

    def forward(self, x, t, caption_embeddings, mask=None):
        # x: (batch_size, in_channels, height, width)
        # t: (batch_size, 1)
        # caption_embeddings: (batch_size, caption_embed_dim)
        # mask: (batch_size, num_patches)
        
        batch_size, channels, height, width = x.shape

        patch_size_h, patch_size_w = self.patch_size

        # Image processing
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        
        # Generate positional embeddings
        # (height // patch_size_h, width // patch_size_w, pos_embed_dim)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed_dim, height // patch_size_h, width // patch_size_w)
        pos_embed = pos_embed.to(x.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        if self.embed_cat:
            x = torch.cat([x, pos_embed], dim=-1)
        else:
            x = x + pos_embed
        
        # Timestep embedding
        t_emb = self.time_embed(t)  # (batch_size, timestep_caption_embed_dim)

        # Caption embedding
        c_emb = self.caption_embed(caption_embeddings)  # (batch_size, timestep_caption_embed_dim)

        mha_out = self.mha(t_emb.unsqueeze(1), c_emb.unsqueeze(1), c_emb.unsqueeze(1))[0].squeeze(1)
        mlp_out = self.mlp(mha_out)
        
        # Pool + MLP
        pool_out = self.pool_mlp(mlp_out.unsqueeze(2))

        # Pool + MLP + t_emb
        pool_out = (pool_out + t_emb).unsqueeze(1)
        
        # Apply linear layer
        cond_signal = self.linear(mlp_out).unsqueeze(1)  # (batch_size, 1, timestep_caption_embed_dim)
        cond = (cond_signal + pool_out).expand(-1, x.shape[1], -1)
        
        # Add conditioning signal to all patches
        # (batch_size, num_patches, embed_dim + pos_embed_dim) -> (batch_size, num_patches, total_embed_dim)
        if self.embed_cat:
            x = torch.cat([x, cond], dim=-1)
        else:
            x = x + cond

        # Patch-mixer
        x = self.patch_mixer(x)

        # Remove masked patches
        if mask is not None:
            x = remove_masked_patches(x, mask)

        # MHA + MLP + Pool + MLP + t_emb
        cond = (mlp_out.unsqueeze(1) + pool_out).expand(-1, x.shape[1], -1)

        if self.embed_cat:
            x[:, :, -self.timestep_caption_embed_dim:] = x[:, :, -self.timestep_caption_embed_dim:] + cond
        else:
            x = x + cond

        # Backbone transformer model
        x = self.backbone(x, c_emb)
        
        # Final output layer
        # (bs, unmasked_num_patches, total_embed_dim) -> (bs, unmasked_num_patches, patch_size_h * patch_size_w * in_channels)
        x = self.output(x)

        # Add masked patches
        if mask is not None:
            # (bs, unmasked_num_patches, patch_size_h * patch_size_w * in_channels) -> (bs, num_patches, patch_size_h * patch_size_w * in_channels)
            x = add_masked_patches(x, mask)

        x = unpatchify(x, self.patch_size, height, width)
        
        return x
    
class CustomDataset(IterableDataset):
    def __init__(self, dataset, index_image_id_map, bucket_manager):
        self.dataset = dataset
        self.index_image_id_map = index_image_id_map
        self.bucket_manager = bucket_manager

    def __iter__(self):
        # Generate batches using the BucketManager
        for batch_data, resolution in self.bucket_manager.generator():
            # batch_data is a list of image_ids
            # resolution is the target resolution for the batch

            # Map image_ids to dataset indices
            indices = [self.index_image_id_map[f"{int(image_id):08d}"] for image_id in batch_data]

            # Fetch data samples from the dataset using indices
            batch_items = [self.dataset[int(idx)] for idx in indices]

            # Prepare batch data
            batch_latents = []
            batch_embeddings = []
            batch_captions = []

            for item in batch_items:
                # Convert 'latent' and 'embedding' to tensors
                latent = torch.tensor(item["latent"]).reshape(4, resolution[0], resolution[1])
                embedding = torch.tensor(item["embedding"])
                caption = item["caption"]

                batch_latents.append(latent)
                batch_embeddings.append(embedding)
                batch_captions.append(caption)

            # Stack tensors (assuming all latents and embeddings are of the same shape)
            batch_latents = torch.stack(batch_latents)
            batch_embeddings = torch.stack(batch_embeddings)

            # Yield the batch data and resolution
            yield {
                "latents": batch_latents,
                "embeddings": batch_embeddings,
                "captions": batch_captions,
                "resolution": resolution,
            }

    def __len__(self):
        # Return the total number of samples (approximate if necessary)
        return self.bucket_manager.batch_total

class LitMicroDiT(L.LightningModule):
    def __init__(self, model, batch_size=1, seed=0, learning_rate=1e-4,
                ln=True, mask_ratio=0.5, dataset_path=f"{DS_DIR_BASE}/{DATASET_NAME}",
                bucket_file = f"{DS_DIR_BASE}/{METADATA_DATASET_NAME}/res_map.json",
                index_image_id_map_path=f"{DS_DIR_BASE}/{METADATA_DATASET_NAME}/image_id_map.json"):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.ln = ln
        self.mask_ratio = mask_ratio
        self.batch_size = batch_size
        self.seed = seed
        self.bucket_manager = None
        self.total_steps = None
        self.dataset_path = dataset_path
        self.bucket_file = bucket_file
        self.index_image_id_map_path = index_image_id_map_path

    def setup(self, stage=None):
        # Initialize the BucketManager here
        self.bucket_manager = BucketManager(
            bucket_file=self.bucket_file,
            bsz=self.batch_size,
            world_size=self.trainer.world_size,
            global_rank=self.global_rank,
            seed=self.seed,
            divisible=2,
            min_dim=32,
            max_size=(64,64),
            base_res=(64,64),
            dim_limit=128
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=self.bucket_manager.batch_total,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def forward(self, x, t, mask):
        self.model(x, t, mask)

    def train_dataloader(self):
        # Load the dataset
        dataset = load_dataset(
            self.dataset_path,
            split="train",
            keep_in_memory=False,
            cache_dir=None,
        )

        # Load the image_id to index mapping
        with open(self.index_image_id_map_path, "r") as f:
            index_image_id_map = json.load(f)

        # Create the CustomDataset, pass the bucket_manager
        dataset = CustomDataset(
            dataset=dataset,
            index_image_id_map=index_image_id_map,
            bucket_manager=self.bucket_manager,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=None,  # Batching is handled internally by the dataset
            num_workers=0,    # Set to 0 to avoid multiprocessing issues
        )
        return dataloader

    def training_step(self, batch, batch_idx):
        latents = batch["latents"]
        caption_embeddings = batch["embeddings"]
        captions = batch["captions"]
        resolution = batch["resolution"]

        bs = latents.shape[0]

        latents = latents * VAE_SCALING_FACTOR
        latents = latents.reshape(bs, 4, resolution[0], resolution[1])

        mask = random_mask(bs, latents.shape[-2], latents.shape[-1], self.model.patch_size, mask_ratio=self.mask_ratio).to(self.device)

        if self.ln:
            nt = torch.randn((bs,)).to(self.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((bs,)).to(self.device)
        texp = t.view([bs, *([1] * len(latents.shape[1:]))])
        z1 = torch.randn_like(latents)
        zt = (1 - texp) * latents + texp * z1
        
        vtheta = self.model(zt, t, caption_embeddings, mask)

        latents = apply_mask_to_tensor(latents, mask, self.model.patch_size)
        vtheta = apply_mask_to_tensor(vtheta, mask, self.model.patch_size)
        z1 = apply_mask_to_tensor(z1, mask, self.model.patch_size)

        batchwise_mse = ((z1 - latents - vtheta) ** 2).mean(dim=list(range(1, len(latents.shape))))
        loss = batchwise_mse.mean()
        loss = loss * 1 / (1 - self.mask_ratio)

        self.log("Loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    @torch.inference_mode()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond, None)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return (images[-1] / VAE_SCALING_FACTOR)