import torch
from diffusers import AutoencoderKL
from transformer.microdit import MicroDiT
from accelerate import Accelerator
from config import BS, EPOCHS, MASK_RATIO, VAE_SCALING_FACTOR, VAE_CHANNELS, VAE_HF_NAME, MODELS_DIR_BASE, SEED
from config import DIT_S as DIT
from torch.amp import autocast
import datasets
from dataset.packing_dataset import get_datasets, CombinedDataset
from torch.utils.data import DataLoader
from transformer.utils import random_mask, apply_mask_to_tensor

def get_dataset(self):
    datasets = get_datasets()
    dataloaders = [DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True) for dataset in datasets]
    dataset = CombinedDataset(dataloaders, self.seed)

    return dataset

if __name__ == "__main__":
    # Comment this out if you havent downloaded dataset and models yet
    datasets.config.HF_HUB_OFFLINE = 1

    input_dim = VAE_CHANNELS  # 4 channels in latent space
    patch_size = (2, 2)
    embed_dim = DIT["embed_dim"]
    num_layers = DIT["num_layers"]
    num_heads = DIT["num_heads"]
    mlp_dim = embed_dim * 4
    caption_embed_dim = 1152  # SigLip embeds to 1152 dims
    # pos_embed_dim = 60
    pos_embed_dim = None
    num_experts = 8
    active_experts = 2
    patch_mixer_layers = 1
    dropout = 0.1

    device = "cuda"

    vae = AutoencoderKL.from_pretrained(f"{VAE_HF_NAME}", cache_dir=f"{MODELS_DIR_BASE}/vae")

    model = MicroDiT(input_dim, patch_size, embed_dim, num_layers, 
                    num_heads, mlp_dim, caption_embed_dim,
                    num_experts, active_experts,
                    dropout, patch_mixer_layers)

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    print("Starting training...")

    accelerator = Accelerator()
    dataset = get_dataset(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, dataset)

    with autocast(device_type="cuda", dtype=torch.float16):
        for epoch in range(EPOCHS):
            for batch in train_dataloader:
                optimizer.zero_grad()
        
                latents = batch["vae_latent"][0]
                caption_embeddings = batch["text_embedding"][0]
                bs = latents.shape[0]

                latents = latents * VAE_SCALING_FACTOR

                mask = random_mask(bs, latents.shape[-2], latents.shape[-1], patch_size, mask_ratio=MASK_RATIO).to(device)

                if model.ln:
                    nt = torch.randn((bs,)).to(device)
                    t = torch.sigmoid(nt)
                else:
                    t = torch.rand((bs,)).to(device)
                texp = t.view([bs, *([1] * len(latents.shape[1:]))])
                z1 = torch.randn_like(latents)
                zt = (1 - texp) * latents + texp * z1
                
                vtheta = model(zt, t, caption_embeddings, mask)

                latents = apply_mask_to_tensor(latents, mask, patch_size)
                vtheta = apply_mask_to_tensor(vtheta, mask, patch_size)
                z1 = apply_mask_to_tensor(z1, mask, patch_size)

                batchwise_mse = ((z1 - latents - vtheta) ** 2).mean(dim=list(range(1, len(latents.shape))))
                loss = batchwise_mse.mean()
                loss = loss * 1 / (1 - MASK_RATIO)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()

        print(f"Epoch {epoch} complete.")

    print("Training complete.")
