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
from models.eval import compute_metrics, mean_dict_values, append_to_dict, write_to_tb
from models.utils import EarlyStopper
import torch.nn.functional as F
import logging

logger = logging.getLogger("model")

logging.getLogger("model").addHandler(logging.NullHandler())

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


    def evaluate(self, dataloader, n_batches=None):
        self.network.eval()
        total_loss = []
        results_for_eval = {}

        logger.info(f"Evaluating model on {n_batches if n_batches is not None else 'all'} batches")

        for batch_count, batch in enumerate(dataloader):
            if batch is None:
                warnings.warn('Empty batch')
                continue

            if n_batches is not None and batch_count >= n_batches:
                logger.info(f"Breaking evaluation loop after {n_batches} batches")
                break

            for key in batch.keys():
                if key == 'patient_id':
                    continue
                batch[key] = batch[key].to(self.config.device)
            
            # FORWARD
            logits = self.network(batch['x'], batch)
            loss = self.get_model_loss(logits, batch)
            probs = torch.sigmoid(logits).cpu().data.numpy()
            batch_results = dict(probs=probs.tolist(),
                        idx_of_last_y_to_eval=batch['idx_of_last_y_to_eval'].tolist(),
                        y=batch['y'].tolist())
            results_for_eval = append_to_dict(results_for_eval, batch_results)

            total_loss.append(loss.item())

        return np.mean(total_loss), compute_metrics(self.config, results_for_eval)


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

            global_step = 0

            num_batches_per_epoch = min(len(train_dataloader), (self.config.n_batches))

            logger.info(f"Starting training for {self.config.num_epochs} epochs with {num_batches_per_epoch} batches in each.")

            for epoch_id in range(self.config.resume_epoch, self.config.resume_epoch + self.config.num_epochs, 1):
                loss_monitor = []
                results_for_eval = {}
                logger.info(f"Epoch {epoch_id + 1}/{self.config.num_epochs}")

                if stop_run:
                    break

                for batch_count, batch in enumerate(tqdm(train_dataloader, total=num_batches_per_epoch)):
                    if batch_count % 50 == 0:
                        logger.info(f"Batch {batch_count + 1}/{num_batches_per_epoch}")

                    if batch_count > num_batches_per_epoch or stop_run:
                        logger.info("Breaking...")
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
                    batch_results = dict(probs=probs.tolist(),
                             idx_of_last_y_to_eval=batch['idx_of_last_y_to_eval'].tolist(),
                             y=batch['y'].tolist())
                    results_for_eval = append_to_dict(results_for_eval, batch_results)
                    loss_monitor.append(loss.cpu().data.item())

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # monitoring
                    if (batch_count + 1) % self.config.n_train_batches_per_eval == 0:
                        # Previously: Calculated global step as epoch_id * num_batches_per_epoch + batch_count + 1
                        global_step += 1
                        logger.info(f"Logging/evaluating at global step {global_step} (epoch {epoch_id}, batch {batch_count})")

                        write_to_tb(tb_writer, "Train_Loss", np.mean(loss_monitor), global_step)

                        loss_monitor = []

                        # -------------------------
                        # Validation
                        # -------------------------
                        if val_dataloader is not None:
                            loss_val, metrics_val = self.evaluate(val_dataloader, self.config.n_batches_for_eval)

                            write_to_tb(tb_writer, "Val_Loss", loss_val, global_step)
                            # for key, val in metrics_val.items():
                            #     write_to_tb(tb_writer, f"Val_{key}", val, global_step)

                            self.network.train()

                            if early_stopper.early_stop(loss_val): # TODO
                                logger.info('Early stopping')
                                stop_run = True


                global_step += 1
                logger.info(f"Epoch {epoch_id} finished. Evaluating on training set and logging at global step {global_step}")
                if loss_monitor: # Make sure it's not empty
                    write_to_tb(tb_writer, "Train_Loss", np.mean(loss_monitor), global_step)

                metrics = compute_metrics(self.config, results_for_eval)
                for key,val in metrics.items():
                    write_to_tb(tb_writer, f"Train_{key}", val, global_step)

                logger.info("Evaluating on full validation set")
                loss_val_full, metrics_val_full = self.evaluate(val_dataloader)
                write_to_tb(tb_writer, "Val_Loss", loss_val_full, global_step)
                for key, val in metrics_val_full.items():
                    write_to_tb(tb_writer, f"Val_{key}", val, global_step)

                # save model
                checkpoint = {
                    "network": self.network,
                    "optimizer": self.optimizer.state_dict()
                }
                checkpoint_path = os.path.join(self.log_path, 'Epoch_{0:d}_global_step_{1:d}.pt'.format((epoch_id + 1), global_step))
                torch.save(obj=checkpoint, f=checkpoint_path)
                logger.info('State dictionaries are saved into {0:s}'.format(checkpoint_path))

            logger.info('Training is completed.')
        finally:
            logger.info('Close tensorboard summary writer')
            tb_writer.close()
