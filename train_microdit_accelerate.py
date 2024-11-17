import torch
from diffusers import AutoencoderKL
from transformer.microdit import MicroDiT
from accelerate import Accelerator
from config import BS, EPOCHS, MASK_RATIO, VAE_SCALING_FACTOR, VAE_CHANNELS, VAE_HF_NAME, MODELS_DIR_BASE, DS_DIR_BASE, SEED, USERNAME, DATASET_NAME
from config import DIT_B as DIT
from datasets import load_dataset
from dataset.shapebatching_dataset import ShapeBatchingDataset
from transformer.utils import random_mask, apply_mask_to_tensor, num_patches
from tqdm import tqdm
import datasets
import torchvision
import os
import pickle

def sample_images(model, vae, noise, embeddings):
    # Use the stored embeddings
    sampled_latents = sample(model, noise, embeddings)
    
    # Decode latents to images
    sampled_images = vae.decode(sampled_latents).sample

    # Log the sampled images
    grid = torchvision.utils.make_grid(sampled_images, nrow=3, normalize=True, scale_each=True)
    return grid

def get_dataset(bs, seed, num_workers=16):
    dataset = load_dataset(f"{USERNAME}/{DATASET_NAME}", split="train", streaming=True).shuffle(seed, buffer_size = bs * 20)
    dataset = ShapeBatchingDataset(dataset, bs, True, seed)
    return dataset

@torch.no_grad()
def sample(model, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
    b = z.size(0)
    dt = 1.0 / sample_steps
    dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
    images = [z]
    n_patches = num_patches(z, patch_size)
    for i in range(sample_steps, 0, -1):
        t = i / sample_steps
        t = torch.tensor(t).to(z.device).to(torch.float16).repeat(b, n_patches)

        vc = model(z, t, cond, None)
        if null_cond is not None:
            vu = model(z, t, null_cond)
            vc = vu + cfg * (vc - vu)

        z = z - dt * vc
        images.append(z)
    return (images[-1] / VAE_SCALING_FACTOR)

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

    accelerator = Accelerator()
    device = accelerator.device

    vae = AutoencoderKL.from_pretrained(f"{VAE_HF_NAME}", cache_dir=f"{MODELS_DIR_BASE}/vae").to(device)

    model = MicroDiT(input_dim, patch_size, embed_dim, num_layers, 
                    num_heads, mlp_dim, caption_embed_dim,
                    num_experts, active_experts,
                    dropout, patch_mixer_layers)

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    print("Starting training...")
    
    dataset = get_dataset(BS, SEED + accelerator.process_index, num_workers=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, dataset)
    
    if accelerator.is_main_process:
        os.makedirs("logs", exist_ok=True)

        noise = torch.randn(9, 4, 32, 32).to(device)
        example_batch = next(iter(dataset))
        example_embeddings = example_batch["text_embedding"][:9].to(device)
        example_captions = example_batch["caption"][:9]
        example_latents = example_batch["vae_latent"][:9].to(device)
        example_ground_truth = vae.decode(example_latents).sample
        grid = torchvision.utils.make_grid(example_ground_truth, nrow=3, normalize=True, scale_each=True)
        torchvision.utils.save_image(grid, f"logs/example_images.png")

        # Save captions
        with open("logs/example_captions.txt", "w") as f:
            for index, caption in enumerate(example_captions):
                f.write(f"{index}: {caption}\n")

        losses = []

    for epoch in range(EPOCHS):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
    
            latents = batch["vae_latent"].to(device)
            caption_embeddings = batch["text_embedding"].to(device)
            bs = latents.shape[0]

            latents = latents * VAE_SCALING_FACTOR

            mask = random_mask(bs, latents.shape[-2], latents.shape[-1], patch_size, mask_ratio=MASK_RATIO).to(device)

            n_patches = num_patches(latents, patch_size)

            nt = torch.randn((bs, n_patches)).to(device)
            t = torch.sigmoid(nt)
            
            # Expand the t tensor from (bs, num_patches) to (bs, 1, h, w), where h and w are the patch sizes. Duplicate the patch value h*w times.
            texp = t.view(bs, n_patches, 1, 1).repeat(1, 1, patch_size[0], patch_size[1])
            texp = texp.view(bs, 1, latents.shape[-2], latents.shape[-1])

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

            progress_bar.set_postfix(loss=loss.item())
            if accelerator.is_local_main_process:
                losses.append(loss.item())

                if batch_idx % 1000 == 0:
                    grid = sample_images(model, vae, noise, example_embeddings)
                    torchvision.utils.save_image(grid, f"logs/sampled_images_epoch_{epoch}_batch_{batch_idx}.png")

        print(f"Epoch {epoch} complete.")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            model_save_path = f"models/microdit_model_epoch_{epoch}.pt"
            torch.save(unwrapped_model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}.")


    print("Training complete.")

    # Save model in /models
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save losses as a pickle
        with open("logs/losses.pkl", "wb") as f:
            pickle.dump(losses, f)

        unwrapped_model = accelerator.unwrap_model(model)
        model_save_path = "models/pretrained_microdit_model.pt"
        torch.save(unwrapped_model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}.")