import torch.nn as nn
from .embed import PatchEmbed, get_2d_sincos_pos_embed
from .utils import remove_masked_patches, add_masked_patches, unpatchify
from .backbone import TransformerBackbone
from .moedit import TimestepEmbedder
import torch
from config import VAE_SCALING_FACTOR
from datasets import load_dataset

class PatchMixer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

    def initialize_weights(self):
        def _init_transformer_layer(module):
            # Initialize the self-attention layers
            nn.init.xavier_uniform_(module.self_attn.in_proj_weight)
            if module.self_attn.in_proj_bias is not None:
                nn.init.constant_(module.self_attn.in_proj_bias, 0)
            nn.init.xavier_uniform_(module.self_attn.out_proj.weight)
            if module.self_attn.out_proj.bias is not None:
                nn.init.constant_(module.self_attn.out_proj.bias, 0)
            # Initialize the linear layers in the feedforward network
            for lin in [module.linear1, module.linear2]:
                nn.init.xavier_uniform_(lin.weight)
                if lin.bias is not None:
                    nn.init.constant_(lin.bias, 0)
            # Initialize the LayerNorm layers
            for ln in [module.norm1, module.norm2]:
                if ln.weight is not None:
                    nn.init.constant_(ln.weight, 1.0)
                if ln.bias is not None:
                    nn.init.constant_(ln.bias, 0)

        # Initialize each TransformerEncoderLayer
        for layer in self.layers:
            _init_transformer_layer(layer)

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
    def __init__(self, in_channels, patch_size, embed_dim, num_layers, num_heads, mlp_dim, caption_embed_dim,
                 num_experts=4, active_experts=2, dropout=0.1, patch_mixer_layers=2, embed_cat=False):
        super().__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Image processing
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        
        # Timestep embedding
        self.time_embed = TimestepEmbedder(self.embed_dim)
        
        # Caption embedding
        self.caption_embed = nn.Sequential(
            nn.Linear(caption_embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # MHA for timestep and caption
        self.mha = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True)
        
        # MLP for timestep and caption
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # Pool + MLP for (MHA + MLP)
        self.pool_mlp = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # Linear layer after MHA+MLP
        self.linear = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Patch-mixer
        self.patch_mixer = PatchMixer(self.embed_dim, num_heads, patch_mixer_layers)
        
        # Backbone transformer model
        self.backbone = TransformerBackbone(self.embed_dim, self.embed_dim, self.embed_dim, num_layers, num_heads, mlp_dim, 
                                        num_experts, active_experts, dropout)
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, patch_size[0] * patch_size[1] * in_channels)
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize all linear layers and biases
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                # Initialize convolutional layers like linear layers
                nn.init.xavier_uniform_(module.weight.view(module.weight.size(0), -1))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize MultiheadAttention layers
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                # Initialize LayerNorm layers
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.TransformerEncoderLayer):
                # Initialize TransformerEncoderLayer modules
                # Initialize the self-attention layers
                nn.init.xavier_uniform_(module.self_attn.in_proj_weight)
                if module.self_attn.in_proj_bias is not None:
                    nn.init.constant_(module.self_attn.in_proj_bias, 0)
                nn.init.xavier_uniform_(module.self_attn.out_proj.weight)
                if module.self_attn.out_proj.bias is not None:
                    nn.init.constant_(module.self_attn.out_proj.bias, 0)
                # Initialize the linear layers in the feedforward network
                for lin in [module.linear1, module.linear2]:
                    nn.init.xavier_uniform_(lin.weight)
                    if lin.bias is not None:
                        nn.init.constant_(lin.bias, 0)
                # Initialize the LayerNorm layers
                for ln in [module.norm1, module.norm2]:
                    if ln.weight is not None:
                        nn.init.constant_(ln.weight, 1.0)
                    if ln.bias is not None:
                        nn.init.constant_(ln.bias, 0)

        # Apply basic initialization to all modules
        self.apply(_basic_init)

        # [Rest of the initialization code remains the same...]

        # Initialize the patch embedding projection
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.patch_embed.proj.bias is not None:
            nn.init.constant_(self.patch_embed.proj.bias, 0)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        if self.time_embed.mlp[0].bias is not None:
            nn.init.constant_(self.time_embed.mlp[0].bias, 0)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        if self.time_embed.mlp[2].bias is not None:
            nn.init.constant_(self.time_embed.mlp[2].bias, 0)

        # Initialize caption embedding layers
        for layer in self.caption_embed:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # Initialize MLP layers in self.mlp
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # Initialize MLP layers in self.pool_mlp
        for layer in self.pool_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # Initialize the linear layer in self.linear
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

        # Zero-out the last linear layer in the output to ensure initial predictions are zero
        nn.init.constant_(self.output[-1].weight, 0)
        if self.output[-1].bias is not None:
            nn.init.constant_(self.output[-1].bias, 0)

        # Initialize the backbone
        self.backbone.initialize_weights()

        # Initialize the PatchMixer
        self.patch_mixer.initialize_weights()

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
        # (height // patch_size_h, width // patch_size_w, embed_dim)
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, height // patch_size_h, width // patch_size_w)
        pos_embed = pos_embed.to(x.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        x = x + pos_embed
        
        # Timestep embedding
        t_emb = self.time_embed(t)  # (batch_size, embed_dim)

        # Caption embedding
        c_emb = self.caption_embed(caption_embeddings)  # (batch_size, embed_dim)

        mha_out = self.mha(t_emb.unsqueeze(1), c_emb.unsqueeze(1), c_emb.unsqueeze(1))[0].squeeze(1)
        mlp_out = self.mlp(mha_out)
        
        # Pool + MLP
        pool_out = self.pool_mlp(mlp_out.unsqueeze(2))

        # Pool + MLP + t_emb
        pool_out = (pool_out + t_emb).unsqueeze(1)
        
        # Apply linear layer
        cond_signal = self.linear(mlp_out).unsqueeze(1)  # (batch_size, 1, embed_dim)
        cond = (cond_signal + pool_out).expand(-1, x.shape[1], -1)
        
        # Add conditioning signal to all patches
        # (batch_size, num_patches, embed_dim)
        x = x + cond

        # Patch-mixer
        x = self.patch_mixer(x)

        # Remove masked patches
        if mask is not None:
            x = remove_masked_patches(x, mask)

        # MHA + MLP + Pool + MLP + t_emb
        cond = (mlp_out.unsqueeze(1) + pool_out).expand(-1, x.shape[1], -1)

        x = x + cond

        # Backbone transformer model
        x = self.backbone(x, c_emb)
        
        # Final output layer
        # (bs, unmasked_num_patches, embed_dim) -> (bs, unmasked_num_patches, patch_size_h * patch_size_w * in_channels)
        x = self.output(x)

        # Add masked patches
        if mask is not None:
            # (bs, unmasked_num_patches, patch_size_h * patch_size_w * in_channels) -> (bs, num_patches, patch_size_h * patch_size_w * in_channels)
            x = add_masked_patches(x, mask)

        x = unpatchify(x, self.patch_size, height, width)
        
        return x
    
    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device).to(torch.float16)

            vc = self(z, t, cond, None)
            if null_cond is not None:
                vu = self(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return (images[-1] / VAE_SCALING_FACTOR)