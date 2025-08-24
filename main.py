import json
import os
import pickle
import random

import numpy as np
import torch

import copy

from dataset.data_utils import get_dataset_loader
from dataset.dataset import DiseaseProgressionDataset
from pathlib import Path

from consts_and_config import PAD_TOKEN, UNK_TOKEN, Config, GROUP_SPLITS
from build_config import get_data_and_config_from_cmdline

from models.mlp import MLP
from models.model import Model
# from train_model import train_model


if __name__ == "__main__":
    print("CUDA:", torch.cuda.is_available())
    results_path = "out"

    data, config = get_data_and_config_from_cmdline()

    print(f"Starting run with config {config}")

    os.makedirs(config.log_dir / config.run_name, exist_ok=True)
    os.makedirs(config.out_dir, exist_ok=True)

    with open(config.log_dir / config.run_name / "config.json", "w") as f:
        json.dump(config.dict(), f)

    # Create Datasets
    train_dataset = DiseaseProgressionDataset(data, config, GROUP_SPLITS.TRAIN)
    test_dataset = DiseaseProgressionDataset(data, config, GROUP_SPLITS.TEST)
    val_dataset = DiseaseProgressionDataset(data, config, GROUP_SPLITS.VALIDATION)
    print(f"Number of patient for -Train:{len(train_dataset)}, -Dev:{len(val_dataset)}, -Test:{len(test_dataset)}")

    # Set Random Seed
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    train_dataloader = get_dataset_loader(config, train_dataset)
    val_dataloader = get_dataset_loader(config, val_dataset)
    test_dataloader = get_dataset_loader(config, test_dataset)

    # Create model
    network = MLP(config)
    network.to(config.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)

    model = Model(network=network, optimizer=optimizer, config=config)

    model.train(train_dataloader, val_dataloader)

    # import ipdb;ipdb.set_trace()