import warnings
import logging
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import typing
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from consts_and_config import Config
from models.eval import compute_metrics, append_to_dict, write_to_tb
from models.utils import ReduceLROnPlateau


logger = logging.getLogger("model")

logging.getLogger("model").addHandler(logging.NullHandler())

class Model:
    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer,
                 config: Config):
        self.network = network
        self.optimizer = optimizer
        self.config = config

    def get_model_loss(self, logits, batch):
        y_seq = batch['y_seq']
        y_mask = batch['y_mask']
        loss = F.binary_cross_entropy_with_logits(logits, y_seq.float(), weight=y_mask.float(), reduction='sum')\
            / torch.sum(y_mask.float())
        return loss


    def evaluate(self, dataloader, n_batches=None, plot_metrics=False):
        self.network.eval()
        total_loss = []
        results_for_eval = {}

        logger.info(f"Evaluating model on {n_batches if n_batches is not None else 'all'} batches")

        for batch_count, batch in enumerate(dataloader):
            if batch is None:
                logger.warning('Empty batch')
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

        return np.mean(total_loss), compute_metrics(self.config, results_for_eval, plot_metrics)


    def train(self, train_dataloader: torch.utils.data.DataLoader,
              val_dataloader: typing.Optional[torch.utils.data.DataLoader]):
        self.optimizer.zero_grad()

        self.network.train()

        tb_writer = SummaryWriter(
            log_dir=self.config.log_dir
            # purge_step=self.config.resume_epoch * self.config['num_episodes_per_epoch'] // self.config['minibatch_print'] if self.config.resume_epoch > 0 else None
        )

        tuning_metric = 'auc_36'
        best_model_filename = f'val_{tuning_metric}_best_model.pt'

        try:
            stop_run = False

            reduce_lr = ReduceLROnPlateau(optimizer=self.optimizer, log_path=self.config.log_dir, device=self.config.device,
                                          curr_lr=self.config.learning_rate ,metric_to_monitor=f"val_{tuning_metric}", mode="max", lr_decay=self.config.lr_decay,
                                          patience=self.config.reduce_lr_patience, min_delta=self.config.min_delta_checkpoint, best_model_filename=best_model_filename)

            global_step = 0
            num_batches_per_epoch = min(len(train_dataloader), (self.config.n_batches))
            logger.info(f"Starting training for {self.config.num_epochs} epochs with {num_batches_per_epoch} batches in each.")

            for epoch_id in range(self.config.resume_epoch, self.config.resume_epoch + self.config.num_epochs, 1):
                loss_monitor = []
                train_results_for_eval = {}
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
                    train_results_for_eval = append_to_dict(train_results_for_eval, batch_results)
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
                            loss_val, _ = self.evaluate(val_dataloader, self.config.n_batches_for_eval)

                            write_to_tb(tb_writer, "Val_Loss", loss_val, global_step)

                            self.network.train()

                            # if early_stopper.early_stop(loss_val):
                            #     logger.info('Early stopping')
                            #     stop_run = True


                global_step += 1
                logger.info(f"Epoch {epoch_id} finished. Evaluating on training set and logging at global step {global_step}")
                if loss_monitor: # Make sure it's not empty
                    write_to_tb(tb_writer, "Train_Loss", np.mean(loss_monitor), global_step)

                metrics_train = compute_metrics(self.config, train_results_for_eval)
                for key,val in metrics_train.items():
                    write_to_tb(tb_writer, f"Train_{key}", val, global_step)

                logger.info("Evaluating on full validation set")
                loss_val_full, metrics_val_full = self.evaluate(val_dataloader)
                write_to_tb(tb_writer, "Val_Loss", loss_val_full, global_step)
                for key, val in metrics_val_full.items():
                    write_to_tb(tb_writer, f"Val_{key}", val, global_step)

                # save model and optimizer weights and biases at each epoch
                checkpoint = {
                    "epoch": epoch_id,
                    "network_state_dict": self.network.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()
                }
                checkpoint_path = self.config.log_dir / 'Epoch_{0:d}_global_step_{1:d}.pt'.format(epoch_id, global_step)
                torch.save(obj=checkpoint, f=checkpoint_path)
                logger.info(f'State dictionaries are saved into {checkpoint_path}')

                # save the best model if improved, if not check patience and reduce lr when necessary
                lr_monitor = reduce_lr.step(metrics_val_full[tuning_metric], epoch_id, self.network)

                write_to_tb(tb_writer, "lr", lr_monitor, global_step)

                # if early_stopper.early_stop(loss_val_full):
                #     logger.info('Early stopping')
                #     stop_run = True


            logger.info('Training is completed.')
        finally:
            logger.info('Close tensorboard summary writer')
            tb_writer.close()

        return self.config.log_dir / best_model_filename
