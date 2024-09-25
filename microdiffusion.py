import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
import numpy as np
from rf import RF
from accelerate import Accelerator
from transformer.micro_dit import MicroDiT
from transformer.utils import strings_to_tensor, random_mask, remove_masked_patches, add_masked_patches
import matplotlib.pyplot as plt
import os

bs = 16
input_dim = 4  # 4 channels in latent space
patch_size = 1 
embed_dim = 384
num_layers = 12
num_heads = 6
mlp_dim = embed_dim * 4
class_label_dim = 40  # 40 attributes in CelebA dataset
patch_mixer_layers = 1

epochs = 1
mask_ratio = 0.75

train_ds = load_from_disk("../../datasets/CelebA-attrs-latents/train")
# validation_ds = load_from_disk("../../datasets/CelebA-attrs-latents/validation")
# test_ds = load_from_disk("../../datasets/CelebA-attrs-latents/test")

train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
# validation_dl = DataLoader(validation_ds, batch_size=bs, shuffle=True)
# test_dl = DataLoader(test_ds, batch_size=bs, shuffle=True)  

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", cache_dir="../../models/vae")
model = MicroDiT(input_dim, patch_size, embed_dim, num_layers, num_heads, mlp_dim, class_label_dim, patch_mixer_layers=patch_mixer_layers)
rf = RF(model, ln=True)

print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
total_steps = len(train_dl) * epochs
scheduler = OneCycleLR(optimizer, max_lr=5e-4, total_steps=total_steps)

accelerator = Accelerator(mixed_precision="fp16")

model, rf, optimizer, train_dl, scheduler = accelerator.prepare(
    model, rf, optimizer, train_dl, scheduler
)

latents_mean = torch.tensor(-0.57).to(accelerator.device)
latents_std = torch.tensor(6.91).to(accelerator.device)

def train_model(model, rf, train_dl, epochs, mask_ratio, latents_mean, latents_std, patch_size, accelerator, optimizer, lr_scheduler, max_steps=None):
    scaler = GradScaler(device=accelerator.device)

    loss_history = []

    i = 0
    if max_steps is None: 
        pbar = tqdm(total=len(train_dl) * epochs)
    else:
        pbar = tqdm(total=max_steps)
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_dl):
            if max_steps is not None:
                if i >= max_steps:
                    return loss_history
            
            bs = batch["latents"].shape[0]
            image_prompts = strings_to_tensor(batch["prompt_string"]).to(accelerator.device)
            latents = batch["latents"]

            latents = (latents - latents_mean) / latents_std

            mask = random_mask(bs, latents.shape[-2], latents.shape[-1], patch_size, mask_ratio=mask_ratio).to(accelerator.device)
            # masked_noise = noise * mask.unsqueeze(1).view(bs, 1, latents.shape[-2], latents.shape[-1])

            optimizer.zero_grad()

            with autocast(device_type=str(accelerator.device)):
                # pred = model(noised_latents, noise_amt, image_prompts, mask)
                # loss = loss_fn(pred, masked_noise) * 1/(1-mask_ratio)
                loss, _ = rf.forward(latents, image_prompts, mask)
                loss = loss * 1/(1-mask_ratio)

            accelerator.backward(scaler.scale(loss))
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            loss_value = loss.item()
            epoch_loss += loss_value
            loss_history.append(loss_value)

            pbar.set_postfix({'loss': f'{loss_value:.4f}'})
            pbar.update(1)

            i += 1

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(train_dl)

        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    return loss_history

max_steps = None

print("Starting training...")
loss_history = train_model(model, rf, train_dl, epochs, mask_ratio, latents_mean, latents_std, patch_size, accelerator, optimizer, scheduler, max_steps=max_steps)
print("Training complete.")

# Save loss history to a file
with open("loss_history.txt", "w") as f:
    for loss in loss_history:
        f.write(f"{loss}\n")

print("Loss history saved to loss_history.txt")

print("Starting finetuning...")

finetuning_steps = total_steps // 10
mask_ratio = 0.

scheduler = OneCycleLR(optimizer, max_lr=1e-4, total_steps=finetuning_steps)

loss_history = train_model(model, rf, train_dl, epochs, mask_ratio, latents_mean, latents_std, patch_size, accelerator, optimizer, scheduler, max_steps=finetuning_steps)

print("Finetuning complete.")

# Save loss history to a file
with open("finetuning_loss_history.txt", "w") as f:
    for loss in loss_history:
        f.write(f"{loss}\n")

print("Finetuning loss history saved to finetuning_loss_history.txt")

# Create models directory if it doesn't exist
os.makedirs('models/diffusion', exist_ok=True)

# Save the model
torch.save(model.state_dict(), 'models/diffusion/microdiffusion_model.pth')

print("Model saved successfully.")