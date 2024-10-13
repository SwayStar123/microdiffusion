from transformer.microdit import LitMicroDiT
import torch
from diffusers import AutoencoderKL
from transformer.microdit import LitMicroDiT, MicroDiT
import os
import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint
from dataset.bucket_manager import BucketManager
if __name__ == "__main__":
    # preprocess_datasets_main(test=True)
    # index_image_id_map_main()

    bs = 32
    input_dim = 4  # 4 channels in latent space
    patch_size = (2, 2)
    # embed_dim = 1152
    # num_layers = 28
    # num_heads = 16
    embed_dim = 384
    num_layers = 12
    num_heads = 6
    mlp_dim = embed_dim * 4
    caption_embed_dim = 1152  # SigLip embeds to 1152 dims
    # pos_embed_dim = 60
    pos_embed_dim = None
    # timestep_caption_embed_dim = 60
    timestep_caption_embed_dim = None
    num_experts = 8
    active_experts = 2
    patch_mixer_layers = 1
    dropout = 0.1
    embed_cat = False

    epochs = 5
    mask_ratio = 0.75

    world_size = torch.cuda.device_count()

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", cache_dir="../../models/vae")
    model = MicroDiT(input_dim, patch_size, embed_dim, num_layers, 
                    num_heads, mlp_dim, caption_embed_dim, timestep_caption_embed_dim,
                    pos_embed_dim, num_experts, active_experts,
                    dropout, patch_mixer_layers, embed_cat)

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    print("Starting training...")

    model = LitMicroDiT(model, mask_ratio=mask_ratio, batch_size=bs, seed=0)

    checkpoint_callback = ModelCheckpoint(dirpath="models/diffusion/", every_n_epochs=1)

    trainer = L.Trainer(max_epochs=epochs, callbacks=[checkpoint_callback])
    tuner = Tuner(trainer)
    tuner.lr_find(model)

    trainer.fit(model=model)

    print("Training complete.")
