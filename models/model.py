from dataclasses import dataclass
import os
import warnings
import numpy as np
import torch.nn as nn
import torch
import typing
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from consts_and_config import Config
from models.eval import compute_metrics, mean_dict_values, merge_duplicate_keys_flat
from models.utils import EarlyStopper
import torch.nn.functional as F


class Model:
    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer,
                 config: Config):
        self.network = network
        self.optimizer = optimizer
        self.config = config
        self.log_path = config.log_dir / config.run_name


    def get_model_loss(self, logits, batch):
        y_seq = batch['y_seq']
        y_mask = batch['y_mask']
        loss = F.binary_cross_entropy_with_logits(logits, y_seq.float(), weight=y_mask.float(), reduction='sum')\
            / torch.sum(y_mask.float())
        return loss


    def evaluate(self, dataloader):
        self.network.eval()
        total_loss = []
        total_metrics = {}

        for batch_count, batch in enumerate(dataloader):
            if batch is None:
                warnings.warn('Empty batch')
                continue

            for key in batch.keys():
                if key == 'patient_id':
                    continue
                batch[key] = batch[key].to(self.config.device)
            
            # FORWARD
            logits = self.network(batch['x'], batch)
            loss = self.get_model_loss(logits, batch)
            probs = torch.sigmoid(logits).cpu().data.numpy()
            metrics = compute_metrics(self.config, probs, batch)
            total_metrics = merge_duplicate_keys_flat(metrics, total_metrics)
            total_loss.append(loss.item())

        return np.mean(total_loss), mean_dict_values(total_metrics)


    def train(self, train_dataloader: torch.utils.data.DataLoader,
              val_dataloader: typing.Optional[torch.utils.data.DataLoader]):
        self.optimizer.zero_grad()

        self.network.train()

        tb_writer = SummaryWriter(
            log_dir=self.log_path
            # purge_step=self.config.resume_epoch * self.config['num_episodes_per_epoch'] // self.config['minibatch_print'] if self.config.resume_epoch > 0 else None
        )

        try:
            stop_run = False
            early_stopper = EarlyStopper(patience=20, min_delta=10)

            global_step_monitor = 0

            num_batches_per_epoch = min(len(train_dataloader), (self.config.n_batches))

            for epoch_id in range(self.config.resume_epoch, self.config.resume_epoch + self.config.num_epochs, 1):
                loss_monitor = []
                metric_monitor = {}

                if stop_run:
                    break

                for batch_count, batch in enumerate(tqdm(train_dataloader, total=num_batches_per_epoch)):

                    if batch_count > num_batches_per_epoch or stop_run:
                        break

                    if batch is None:
                        warnings.warn('Empty batch')
                        continue
                    

                    for key in batch.keys():
                        if key == 'patient_id':
                            continue
                        batch[key] = batch[key].to(self.config.device)
                    
                    # FORWARD
                    logits = self.network(batch['x'], batch)
                    loss = self.get_model_loss(logits, batch)

                    probs = torch.sigmoid(logits).cpu().data.numpy()
                    metrics = compute_metrics(self.config, probs, batch)
                    loss_monitor.append(loss.cpu().data.item())
                    metric_monitor = merge_duplicate_keys_flat(metrics, metric_monitor)

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # monitoring
                    if (batch_count + 1) % self.config.n_batches_per_eval == 0:
                        # calculate step for Tensorboard Summary Writer
                        global_step = (epoch_id * num_batches_per_epoch + batch_count + 1) // self.config.n_batches_per_eval
                        global_step_monitor = global_step
                        
                        mean_dict = mean_dict_values(metric_monitor)

                        tb_writer.add_scalar(tag="Train_Loss", scalar_value=np.mean(loss_monitor), global_step=global_step)
                        for key,val in mean_dict.items():
                            tb_writer.add_scalar(tag=f"Train_{key}", scalar_value=val, global_step=global_step)

                        loss_monitor = []
                        metric_monitor = {}

                        # -------------------------
                        # Validation
                        # -------------------------
                        if val_dataloader is not None:
                            loss_val, metrics_val = self.evaluate(val_dataloader)

                            tb_writer.add_scalar(tag="Val_loss", scalar_value=loss_val, global_step=global_step)
                            for key, val in metrics_val.items():
                                tb_writer.add_scalar(tag=f"Val_{key}", scalar_value=val, global_step=global_step)

                            self.network.train()

                            if early_stopper.early_stop(loss_val): # TODO
                                print('Early stopping')
                                stop_run = True

                # save model
                checkpoint = {
                    "network": self.network,
                    "optimizer": self.optimizer.state_dict()
                }
                checkpoint_path = os.path.join(self.log_path, 'Epoch_{0:d}_global_step_{1:d}.pt'.format((epoch_id + 1), global_step_monitor))
                torch.save(obj=checkpoint, f=checkpoint_path)
                print('State dictionaries are saved into {0:s}\n'.format(checkpoint_path))

            print('Training is completed.')
        finally:
            print('\nClose tensorboard summary writer')
            tb_writer.close()
