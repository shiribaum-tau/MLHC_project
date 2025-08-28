import json
import logging
import os
import pickle
import random

import numpy as np
import torch

from dataset.data_utils import get_dataset_loader
from dataset.dataset import DiseaseProgressionDataset
from pathlib import Path

from consts_and_config import PAD_TOKEN, UNK_TOKEN, Config, GROUP_SPLITS
from build_config import get_data_and_config_from_cmdline
from models.eval import plot_multi_roc_pr

from models.mlp import MLP
from models.model import Model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    logging.info("CUDA: %s", torch.cuda.is_available())
    results_path = "out"

    data, config = get_data_and_config_from_cmdline()

    logging.info("Running on device %s", config.device)

    logging.info("Starting run with config %s", config)

    os.makedirs(config.log_dir / config.run_name, exist_ok=True)
    os.makedirs(config.out_dir, exist_ok=True)

    with open(config.log_dir / config.run_name / "config.json", "w") as f:
        json.dump(config.dict(), f)

    # Create Datasets
    train_dataset = DiseaseProgressionDataset(data, config, GROUP_SPLITS.TRAIN)
    test_dataset = DiseaseProgressionDataset(data, config, GROUP_SPLITS.TEST)
    val_dataset = DiseaseProgressionDataset(data, config, GROUP_SPLITS.VALIDATION)
    logging.info("Number of patient for -Train:%d, -Dev:%d, -Test:%d", len(train_dataset), len(val_dataset), len(test_dataset))

    # Set Random Seed
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    train_dataloader = get_dataset_loader(config, train_dataset)
    val_dataloader = get_dataset_loader(config, val_dataset)
    test_dataloader = get_dataset_loader(config, test_dataset)
    best_model_path = None

    if config.train:
        # Create model
        network = MLP(config)
        network.to(config.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)

        model = Model(network=network, optimizer=optimizer, config=config)

        best_model_path = model.train(train_dataloader, val_dataloader)

    if config.val:
        network = MLP(config)
        network.to(config.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)

        model = Model(network=network, optimizer=optimizer, config=config)

        if config.train:
            model_path_to_load = best_model_path
        else:
            model_path_to_load = config.model_to_load

        saved_checkpoint = torch.load(
            f=model_path_to_load,
            map_location=
            lambda storage, loc: storage.cuda(config.device.index) if config.device.type == 'cuda' else storage
        )

        # load state dictionaries
        model.network.load_state_dict(state_dict=saved_checkpoint['network_state_dict'])
        model.optimizer.load_state_dict(state_dict=saved_checkpoint['optimizer_state_dict'])

        val_loss, metrics_full = model.evaluate(val_dataloader, plot_metrics=True)

        plot_multi_roc_pr(
            metrics=metrics_full,
            endpoints=config.month_endpoints,
            save_dir=config.out_dir
        )