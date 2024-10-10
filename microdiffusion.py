import torch
from dataset.celebaattrs import CelebAAttrsDataset
from diffusers import AutoencoderKL
from transformer.microdit import LitMicroDiT, MicroDiT
import os
import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import StochasticWeightAveraging, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

bs = 128
input_dim = 4  # 4 channels in latent space
patch_size = (2, 2)
embed_dim = 384
num_layers = 12
num_heads = 6
mlp_dim = embed_dim * 4
class_label_dim = 40  # 40 attributes in CelebA dataset
pos_embed_dim = 60
timestep_class_embed_dim = 60
num_experts = 4
active_experts = 2
patch_mixer_layers = 1
dropout = 0.1
embed_cat = True

epochs = 5
mask_ratio = 0.75

train_ds = CelebAAttrsDataset("train")
# validation_ds = CelebAAttrsDataset("validation")
# test_ds = CelebAAttrsDataset("test")

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", cache_dir="../../models/vae")
model = MicroDiT(input_dim, patch_size, embed_dim, num_layers, 
                num_heads, mlp_dim, class_label_dim, timestep_class_embed_dim,
                pos_embed_dim, num_experts, active_experts,
                dropout, patch_mixer_layers, embed_cat)

print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

print("Starting training...")

model = LitMicroDiT(model, mask_ratio=mask_ratio, batch_size=bs, train_ds=train_ds)

checkpoint_callback = ModelCheckpoint(dirpath="models/diffusion/", every_n_epochs=5)
# swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)

trainer = L.Trainer(max_epochs=epochs, callbacks=[checkpoint_callback])
tuner = Tuner(trainer)
#tuner.scale_batch_size(model, mode="power")
tuner.lr_find(model)

trainer.fit(model=model)

print("Training complete.")


print("Starting finetuning...")

finetuning_steps = model.trainer.estimated_stepping_batches * bs // 10
model.batch_size = int(bs * (1-mask_ratio) * 0.5)
finetuning_steps = finetuning_steps // model.batch_size
model.mask_ratio = 0

trainer = L.Trainer(max_steps=finetuning_steps, callbacks=[checkpoint_callback])
tuner = Tuner(trainer)
# tuner.scale_batch_size(model, mode="power")
tuner.lr_find(model)

trainer.fit(model=model)

print("Finetuning complete.")

# Create models directory if it doesn't exist
os.makedirs('models/diffusion', exist_ok=True)

# Save the model
torch.save(model.state_dict(), 'models/diffusion/microdiffusion_model.pth')

print("Model saved successfully.")