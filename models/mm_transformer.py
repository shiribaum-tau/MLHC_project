import math
from igraph import config
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import Transformer


class MMTransformer(Transformer):
    def __init__(self, config):
        super(MMTransformer, self).__init__(config)
        self.n_types = len(config.token_types)
        self.use_numerical_input = config.use_numerical_input
        self.type_embed = nn.Embedding(self.n_types, config.hidden_dim, padding_idx=0)
        if self.use_numerical_input:
            self.numerical_embed = nn.Linear(1, config.hidden_dim)

    def get_x_embedding_for_transformer(self, embed_x, batch):
        if self.use_numerical_input:
            # embed_x contains the categorical embedding
            num_emb = self.numerical_embed(batch['x'])
            is_categorical = batch['is_categorical_seq']
            # Masking: each token gets either type_emb or num_emb
            embed_x = is_categorical * embed_x + (1 - is_categorical) * num_emb

        type_embed = self.type_embed(batch['type_seq'])
        embed_x = embed_x + type_embed
        return self.condition_on_pos_embed(embed_x, batch)
