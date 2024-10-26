import torch
from diffusers import AutoencoderKL
from transformer.microdit import LitMicroDiT, MicroDiT
import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint
from config import BS, EPOCHS, MASK_RATIO, VAE_CHANNELS, VAE_HF_NAME, MODELS_DIR_BASE, SEED
from config import DIT_S as DIT
from torch.amp import autocast
import datasets

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

    vae = AutoencoderKL.from_pretrained(f"{VAE_HF_NAME}", cache_dir=f"{MODELS_DIR_BASE}/vae")

    model = MicroDiT(input_dim, patch_size, embed_dim, num_layers, 
                    num_heads, mlp_dim, caption_embed_dim,
                    num_experts, active_experts,
                    dropout, patch_mixer_layers)

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    print("Starting training...")

    with autocast(device_type="cuda", dtype=torch.float16):
        model = LitMicroDiT(model, vae=vae, epochs=EPOCHS, batch_size=BS, num_workers=16, seed=SEED, mask_ratio=MASK_RATIO)

        checkpoint_callback = ModelCheckpoint(dirpath="models/diffusion/", every_n_epochs=1)

        trainer = L.Trainer(max_epochs=EPOCHS, callbacks=[checkpoint_callback])
        tuner = Tuner(trainer)
        tuner.lr_find(model)

        trainer.fit(model=model)

    print("Training complete.")
