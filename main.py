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
from models.eval import plot_multi_roc_pr

from models.mlp import MLP
from models.model import Model


logging.basicConfig(
    level=logging.INFO,
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
    best_model_path = None

    if config.train:
        # Create model
        network = MLP(config)
        network.to(config.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)

        model = Model(network=network, optimizer=optimizer, config=config)

        best_model_path = model.train(train_dataloader, val_dataloader)


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
