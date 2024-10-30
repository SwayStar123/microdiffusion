import torch
from diffusers import AutoencoderKL
from transformer.microdit import MicroDiT
from accelerate import Accelerator
from config import BS, EPOCHS, MASK_RATIO, VAE_SCALING_FACTOR, VAE_CHANNELS, VAE_HF_NAME, MODELS_DIR_BASE, DS_DIR_BASE, SEED, USERNAME, DATASET_NAME
from config import DIT_S as DIT
from datasets import load_dataset
from dataset.shapebatching_dataset import Coco30kShapeBatchingDataset
from transformer.utils import random_mask, apply_mask_to_tensor
from tqdm import tqdm

def get_dataset(bs, seed, num_workers=16):
    dataset = load_dataset(f"{USERNAME}/{DATASET_NAME}", cache_dir=f"{DS_DIR_BASE}/{DATASET_NAME}", split="train").to_iterable_dataset(100).shuffle(seed, buffer_size = bs * 20)
    dataset = Coco30kShapeBatchingDataset(dataset, bs, True, seed)
    return dataset

if __name__ == "__main__":
    # Comment this out if you havent downloaded dataset and models yet
    # datasets.config.HF_HUB_OFFLINE = 1

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
    dataset = get_dataset(BS, SEED, num_workers=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, dataset)

    for epoch in range(EPOCHS):
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
    
            latents = batch["vae_latent"].to(device)
            caption_embeddings = batch["text_embedding"].to(device)
            bs = latents.shape[0]

            latents = latents * VAE_SCALING_FACTOR

            mask = random_mask(bs, latents.shape[-2], latents.shape[-1], patch_size, mask_ratio=MASK_RATIO).to(device)

            nt = torch.randn((bs,)).to(device)
            t = torch.sigmoid(nt)
            
            texp = t.view([bs, *([1] * len(latents.shape[1:]))]).to(device)
            z1 = torch.randn_like(latents, device=device)
            zt = (1 - texp) * latents + texp * z1
            
            vtheta = model(zt, t, caption_embeddings, mask)

            latents = apply_mask_to_tensor(latents, mask, patch_size)
            vtheta = apply_mask_to_tensor(vtheta, mask, patch_size)
            z1 = apply_mask_to_tensor(z1, mask, patch_size)

            batchwise_mse = ((z1 - latents - vtheta) ** 2).mean(dim=list(range(1, len(latents.shape))))
            loss = batchwise_mse.mean()
            loss = loss * 1 / (1 - MASK_RATIO)

            accelerator.backward(loss)
            optimizer.step()

        print(f"Epoch {epoch} complete.")

    print("Training complete.")

    # Save model in /models
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    model_save_path = "models/microdit_model.pt"
    torch.save(unwrapped_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}.")