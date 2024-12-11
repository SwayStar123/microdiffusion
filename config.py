USERNAME = "SwayStar123"
DATASET_NAME = "FFHQ_1024_DC-AE_f32"
DS_DIR_BASE = "../../datasets"
MODELS_DIR_BASE = "../../models"
VAE_SCALING_FACTOR = 0.13025

BS = 32
EPOCHS = 1
MASK_RATIO = 0.75
SEED = 42

LR = 1e-4

VAE_HF_NAME = "dc-ae-f32c32-mix-1.0"
VAE_CHANNELS = 32
SIGLIP_HF_NAME = "hf-hub:timm/ViT-SO400M-14-SigLIP-384"
SIGLIP_EMBED_DIM = 1152

DIT_G = dict(
    num_layers=40,
    num_heads=16,
    embed_dim=1408,
)
DIT_XL = dict(
    num_layers=28,
    num_heads=16,
    embed_dim=1152,
)
DIT_L = dict(
    num_layers=24,
    num_heads=16,
    embed_dim=1024,
)
DIT_B = dict(
    num_layers=12,
    num_heads=12,
    embed_dim=768,
)
DIT_S = dict(
    num_layers=12,
    num_heads=6,
    embed_dim=384,
)
