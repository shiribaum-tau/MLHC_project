from itertools import product
import json
import logging
import os
import random
from dataclasses import replace

import numpy as np
import torch


from dataset.data_utils import get_dataset_loader
from dataset.dataset import DiseaseProgressionDataset

from consts_and_config import GROUP_SPLITS
from build_config import get_data_and_config_from_cmdline

from models.mlp import MLP
from models.model import Model
# from train_model import train_model

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

def grid_search(param_grid, base_config, data):
    keys, values = zip(*param_grid.items())
    results = []
    for v in product(*values):
        params = dict(zip(keys, v))
        # Update config for this run
        new_config = params
        new_config['run_name'] = f"{base_config.run_name}_grid_{'__'.join(f'{k}_{val}' for k, val in params.items())}"
        # Set run-specific log_dir
        new_config['log_dir'] = base_config.log_dir / new_config['run_name']
        print(new_config)
        config = replace(base_config, **new_config)
        os.makedirs(config.log_dir, exist_ok=True)
        with open(config.log_dir / "config.json", "w") as f:
            json.dump(config.dict(), f)

        single_job(config, data)

        # Optionally collect results
        # results.append({**params, 'metrics': ...})
    # Optionally, save results to a file
    # with open('grid_search_results.json', 'w') as f:
    #     json.dump(results, f)

def single_job(config, data):
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
    # test_dataloader = get_dataset_loader(config, test_dataset)

    # Create model
    network = MLP(config)
    network.to(config.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)

    model = Model(network=network, optimizer=optimizer, config=config)

    model.train(train_dataloader, val_dataloader)


if __name__ == "__main__":
    logging.info("CUDA: %s", torch.cuda.is_available())

    data, config = get_data_and_config_from_cmdline()

    logging.info("Running on device %s", config.device)

    logging.info("Starting run with config %s", config)

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.out_dir, exist_ok=True)

    with open(config.log_dir / "base_config.json", "w") as f:
        json.dump(config.dict(), f)

        # Example grid
    param_grid = {
        'learning_rate': [0.001, 0.0005],
        # 'hidden_dim': [24, 48],
        'dropout': [0.2, 0.5]
    }

    # Run grid search
    grid_search(param_grid, config, data)
    # import ipdb;ipdb.set_trace()