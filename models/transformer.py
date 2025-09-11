import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.abstract_risk_model import AbstractRiskModel


class Transformer(AbstractRiskModel):
    def __init__(self, config):
        super(Transformer, self).__init__(config)
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.use_time_embed = config.use_time_embed
        self.use_age_embed = config.use_age_embed

        # Time and age embedding layers (if used)
        if config.use_time_embed:
            self.t_embed_add_fc = nn.Linear(config.time_embed_dim, config.hidden_dim)
            self.t_embed_scale_fc = nn.Linear(config.time_embed_dim, config.hidden_dim)
        if config.use_age_embed:
            self.a_embed_add_fc = nn.Linear(config.time_embed_dim, config.hidden_dim)
            self.a_embed_scale_fc = nn.Linear(config.time_embed_dim, config.hidden_dim)

        for layer in range(self.num_layers):
            transformer_layer = TransformerLayer(config)
            self.add_module('transformer_layer_{}'.format(layer), transformer_layer)


    # def condition_on_pos_embed(self, x, embed, embed_type='time'):
    #     if embed_type == 'time':
    #         return self.t_embed_scale_fc(embed) * x + self.t_embed_add_fc(embed)
    #     elif embed_type == 'age':
    #         return self.a_embed_scale_fc(embed) * x + self.a_embed_add_fc(embed)
    #     else:
    #         raise NotImplementedError("Embed type {} not supported".format(embed_type))

    def condition_on_pos_embed(self, embed_x, batch):
        if self.use_time_embed:
            time = batch['time_seq'].float()
            # embed_x = self.condition_on_pos_embed(embed_x, time, 'time')
            embed_x = self.t_embed_scale_fc(time) * embed_x + self.t_embed_add_fc(time)

        if self.use_age_embed:
            age = batch['age_seq'].float()
            # embed_x = self.condition_on_pos_embed(embed_x, age, 'age')
            embed_x = self.a_embed_scale_fc(age) * embed_x + self.a_embed_add_fc(age)

        return embed_x

    def get_x_embedding_for_transformer(self, embed_x, batch):
        return self.condition_on_pos_embed(embed_x, batch)

    def encode_trajectory(self, embed_x, batch=None):
        """
            Computes a forward pass of the model.

            Returns:
                The result of feeding the input through the model.
        """
        embed_x = self.get_x_embedding_for_transformer(embed_x, batch)

        # Run through transformer
        seq_x = embed_x
        for indx in range(self.num_layers):
            name = 'transformer_layer_{}'.format(indx)
            seq_x = self._modules[name](seq_x)
        return seq_x


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super(TransformerLayer, self).__init__()

        self.multihead_attention = MultiHeadAttention(config)
        self.layernorm_attn = nn.LayerNorm(config.hidden_dim)
        self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.layernorm_fc = nn.LayerNorm(config.hidden_dim)

    def forward(self, x):
        h = self.multihead_attention(x)
        x = self.layernorm_attn(h + x)
        h = self.fc2(self.relu(self.fc1(x)))
        x = self.layernorm_fc(h + x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        assert config.hidden_dim % config.num_heads == 0

        self.query = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(p=config.dropout)

        self.dim_per_head = config.hidden_dim // config.num_heads
        self.num_heads = config.num_heads

        self.aggregate_fc = nn.Linear(config.hidden_dim, config.hidden_dim)

    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim_per_head)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, x):
        B, N, H = x.size()

        # perform linear operation and split into h heads
        k = self.key(x).view(B, N, self.num_heads, self.dim_per_head)
        q = self.query(x).view(B, N, self.num_heads, self.dim_per_head)
        v = self.value(x).view(B, N, self.num_heads, self.dim_per_head)

        # transpose to get dimensions B * num_heads * S * dim_per_head
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        h = self.attention(q, k, v)

        # concatenate heads and put through final linear layer
        h = h.transpose(1, 2).contiguous().view(B, -1, H)

        output = self.aggregate_fc(h)
        return output
