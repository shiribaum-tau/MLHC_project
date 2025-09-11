import torch
import torch.nn as nn
import warnings
from models.pooling import GlobalAvgPool
from models.utils import CumulativeProbabilityLayer

from consts_and_config import Config

NAME_TO_POOL = {
    'GlobalAvgPool': GlobalAvgPool
}

class AbstractRiskModel(nn.Module):
    """
        The overall abstract model framework for all model architectures. The model is consists of embedding layers,
        encoding layers, and prediction layers a discrete time survival objective.
    """

    def __init__(self, config: Config):

        super(AbstractRiskModel, self).__init__()

        self.vocab_size = len(config.vocab)
        self.code_embed = nn.Embedding(self.vocab_size, config.hidden_dim, padding_idx=0)
        kept_token_vec = torch.nn.Parameter(torch.ones([1, 1, 1]),  requires_grad=False)
        self.register_parameter('kept_token_vec', kept_token_vec)

        self.pool = NAME_TO_POOL[config.pool_name](config)
        self.dropout = nn.Dropout(p=config.dropout)
        self.prob_of_failure_layer = CumulativeProbabilityLayer(config.hidden_dim, len(config.month_endpoints), config)

    def get_embeddings(self, x, batch=None):
        token_embed = self.code_embed(x)
        return token_embed

    def forward(self, x, batch=None):
        embed_x = self.get_embeddings(x, batch)

        seq_hidden = self.encode_trajectory(embed_x, batch)
        seq_hidden = seq_hidden.transpose(1, 2)
        hidden = self.dropout(self.pool(seq_hidden))
        logit = self.prob_of_failure_layer(hidden)

        return logit
