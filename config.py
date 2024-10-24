USERNAME = "SwayStar123"
DATASET_NAME = "preprocessed_commoncatalog-cc-by"
DS_DIR_BASE = "../../datasets"
MODELS_DIR_BASE = "../../models"
VAE_SCALING_FACTOR = 0.13025

BS = 64
EPOCHS = 5
MASK_RATIO = 0.75
SEED = 42

VAE_HF_NAME = "madebyollin/sdxl-vae-fp16-fix"
VAE_CHANNELS = 4
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
