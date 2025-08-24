import torch
import torch.nn as nn


class CumulativeProbabilityLayer(nn.Module):
    """
        The cumulative layer which defines the monotonically increasing risk scores.
    """
    def __init__(self, num_features, max_followup, args):
        super(CumulativeProbabilityLayer, self).__init__()
        self.args = args
        self.hazard_fc = nn.Linear(num_features,  max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.relu = nn.ReLU(inplace=True)
        mask = torch.ones([max_followup, max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter('upper_triagular_mask', mask)

    def hazards(self, x):
        raw_hazard = self.hazard_fc(x)
        pos_hazard = self.relu(raw_hazard)
        return pos_hazard

    def forward(self, x):
        hazards = self.hazards(x)
        B, T = hazards.size()  # hazards is (B, T)
        expanded_hazards = hazards.unsqueeze(-1).expand(B, T, T)  # expanded_hazards is (B,T, T)
        masked_hazards = expanded_hazards * self.upper_triagular_mask  # masked_hazards now (B,T, T)
        cum_prob = torch.sum(masked_hazards, dim=1) + self.base_hazard_fc(x)
        return cum_prob

class EarlyStopper:
    """
    Keeps track of a chosen metric in order to early stop during training
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_acc = 0

    def early_stop(self, validation_acc):
        if validation_acc > self.max_validation_acc:
            self.max_validation_acc = validation_acc
            self.counter = 0
        elif validation_acc < (self.max_validation_acc - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# class OneHotLayer(nn.Module):
#     """
#         One-hot embedding for categorical inputs.
#     """
#     def __init__(self, num_classes, padding_idx):
#         super(OneHotLayer, self).__init__()
#         self.num_classes = num_classes
#         self.embed = nn.Embedding(num_classes, num_classes, padding_idx=padding_idx)
#         self.embed.weight.data = torch.eye(num_classes)
#         self.embed.weight.requires_grad_(False)

#     def forward(self, x):
#         return self.embed(x)

