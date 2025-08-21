import torch.nn as nn
from models.abstract_risk_model import AbstractRiskModel

class MLP(AbstractRiskModel):
    """
        A basic risk model that embeds codes using a multi-layer perception.
    """
    def __init__(self, config):
        super(MLP, self).__init__(config)
        for layer in range(config.num_layers):
            linear_layer = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.add_module('linear_layer_{}'.format(layer), linear_layer)
        self.relu = nn.ReLU()

    def encode_trajectory(self, embed_x, batch=None):
        seq_hidden = embed_x
        for indx in range(self.config.num_layers):
            name = 'linear_layer_{}'.format(indx)
            seq_hidden = self._modules[name](seq_hidden)
            seq_hidden = self.relu(seq_hidden)
        return seq_hidden
