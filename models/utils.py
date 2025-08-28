import torch
import torch.nn as nn
import os
import logging

logger = logging.getLogger("utils")

logging.getLogger("utils").addHandler(logging.NullHandler())

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
    def __init__(self, patience=1, min_delta=0, metric_to_monitor='val_accuracy', mode='max'):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training stops.
            min_delta (float): Minimum change in monitored value to qualify as an improvement.
            metric_to_monitor (str): Name of the metric being monitored (for logging only).
            mode (str): 'min' for minimizing metric, 'max' for maximizing metric.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric_to_monitor = metric_to_monitor
        self.mode = mode

        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.counter = 0

    def early_stop(self, curr_metric_value):
        if self.mode == 'min':
            if curr_metric_value < self.best_value - self.min_delta:
                self.best_value = curr_metric_value
                self.counter = 0
            else:
                self.counter += 1

        elif self.mode == 'max':
            if curr_metric_value > self.best_value + self.min_delta:
                self.best_value = curr_metric_value
                self.counter = 0
            else:
                self.counter += 1
        else:
            raise ValueError("mode must be either 'min' or 'max'")

        return self.counter >= self.patience


class ModelCheckpoint:
    """
    Saves a copy of the model whenever the monitored metric improves
    """
    def __init__(self, save_dir, metric_to_monitor='val_loss', mode='min', min_delta=0.0, filename='best_model.pt'):
        """
        Args:
            save_dir (str): Directory where to save the best model file.
            metric_to_monitor (str): Name of the metric to monitor (for logging only).
            mode (str): 'min' to minimize the metric, 'max' to maximize.
            min_delta (float): Minimum change to qualify as an improvement.
            filename (str): File name for saving the best model.
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.monitor = metric_to_monitor
        self.mode = mode
        self.min_delta = min_delta
        self.filename = filename

        self.best_value = float('inf') if mode == 'min' else -float('inf')

    def check_and_save(self, metric_value, model, optimizer, epoch_id):
        """
        Checks if the metric improved and saves the model if it did.

        Args:
            metric_value (float): Current value of the monitored metric.
            model (torch.nn.Module): Model to save.
            optimizer (torch.optim.Optimizer): Optimizer state to save.
            epoch_id (int): Epoch number.
        """
        improved = False

        if self.mode == 'min':
            if metric_value < self.best_value - self.min_delta:
                improved = True
        elif self.mode == 'max':
            if metric_value > self.best_value + self.min_delta:
                improved = True

        if improved:
            self.best_value = metric_value
            best_path = os.path.join(self.save_dir, self.filename)
            checkpoint = {
                'epoch': epoch_id,
                'network_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, best_path)
            logger.info(f"Model improved ({self.monitor} = {metric_value:.4f}). Saved to {best_path}")


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

