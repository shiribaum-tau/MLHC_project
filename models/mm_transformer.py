import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import Transformer


class MMTransformer(Transformer):
    """
    Multi-modal Transformer model for handling both categorical and numerical features.
    Inherits from Transformer and adds type/numerical embeddings.
    """
    def __init__(self, config):
        """
        Initializes the MMTransformer.

        Args:
            config: Run configuration object.
        """
        super(MMTransformer, self).__init__(config)
        self.n_types = len(config.token_types)
        self.type_embed = nn.Embedding(self.n_types, config.hidden_dim, padding_idx=0)
        self.numerical_embed = nn.Linear(1, config.hidden_dim)

    def get_embeddings(self, x, batch=None):
        """
        Returns embeddings for categorical codes only, zeroing out non-categorical codes.

        Args:
            x (Tensor): Input codes.
            batch (dict): Batch data including is_categorical_seq.

        Returns:
            Tensor: Embeddings for categorical codes.
        """
        assert x.dim() == 2, f"x should be (B,T), got {x.shape}"
        assert batch['is_categorical_seq'].shape == x.shape

        # the code_embed layer only supports longs. Zero out non-categorical codes
        x_masked = (x * batch['is_categorical_seq']).to(torch.long)
        return super().get_embeddings(x_masked, batch)

    def get_x_embedding_for_transformer(self, embed_x, batch):
        """
        Fuses categorical and numerical embeddings, adds type embeddings, and applies positional conditioning.
        """
        B, T, H = embed_x.shape
        assert batch['x'].shape == (B, T)
        assert batch['type_seq'].shape == (B, T)
        assert batch['is_categorical_seq'].shape == (B, T)

        # Get encoding for type
        type_embed = self.type_embed(batch['type_seq'])

        # embed_x already contains embeddings for categorical codes only
        # Now get embeddings for numerical codes
        x_num = batch['x'].float().unsqueeze(-1)   # (B, T, 1)
        num_emb = self.numerical_embed(x_num)      # (B, T, H)

        # Fuse the two embeddings based on the mask
        mask = batch['is_categorical_seq'].unsqueeze(-1).to(embed_x.dtype)  # (B, T, 1)
        fused = mask * embed_x + (1.0 - mask) * num_emb  

        # Incorporate the type embeddings
        fused = fused + type_embed  # (B, T, H)

        out = self.condition_on_pos_embed(fused, batch)
        assert out.shape == (B, T, H)
        return out