from consts_and_config import GROUP_SPLITS, Config, SUPPORTED_MODELS
from itertools import product
import json
import logging
import os
import random
from dataclasses import replace

import numpy as np
import torch
import pandas as pd

from dataset.data_utils import get_dataset_loader
from dataset.dataset import DiseaseProgressionDataset

from consts_and_config import GROUP_SPLITS, Config
from build_config import get_data_and_config_from_cmdline
from models.eval import plot_metrics

from models.mlp import MLP
from models.mm_transformer import MMTransformer
from models.model import Model
from models.transformer import Transformer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

def create_model_and_optimizer(config):
    if config.model_type == SUPPORTED_MODELS.MLP:
        network = MLP(config)
    elif config.model_type == SUPPORTED_MODELS.TRANSFORMER:
        network = Transformer(config)
    elif config.model_type == SUPPORTED_MODELS.MM_TRANSFORMER:
        network = MMTransformer(config)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
    network.to(config.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    model = Model(network=network, optimizer=optimizer, config=config)
    return model

def grid_search(param_grid, base_config: Config, data):

    os.makedirs(base_config.run_dir, exist_ok=True)
    with open(base_config.run_dir / "base_config.json", "w") as f:
        json.dump(base_config.dict(), f)

    keys, values = zip(*param_grid.items())
    results = []
    csv_path = base_config.run_dir / "grid_search_results.csv"
    for v in product(*values):
        params = dict(zip(keys, v))
        # Update config for this run
        new_config = dict(params)
        new_config['run_name'] = f"{base_config.run_name}_grid_{'__'.join(f'{k}_{val}' for k, val in params.items())}"
        new_config['run_dir'] = base_config.run_dir / new_config['run_name']
        print(new_config)
        config = replace(base_config, **new_config)
        config.train = True
        config.val = True

        out = single_run(config, data)
        for result in out:
            results.append({**params, **result})

        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        logging.info(f"Grid search results saved to {csv_path}")
    

def single_run(config, data):
    # Set Random Seed
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    # Make run directories
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.out_dir, exist_ok=True)

    with open(config.log_dir / "config.json", "w") as f:
        json.dump(config.dict(), f)

    # Create Datasets
    train_dataset = DiseaseProgressionDataset(data, config, GROUP_SPLITS.TRAIN)
    val_dataset = DiseaseProgressionDataset(data, config, GROUP_SPLITS.VALIDATION)

    train_dataloader = get_dataset_loader(config, train_dataset)
    val_dataloader = get_dataset_loader(config, val_dataset)

    if config.test:
        test_dataset = DiseaseProgressionDataset(data, config, GROUP_SPLITS.TEST)
        test_dataloader = get_dataset_loader(config, test_dataset)

    logging.info(f"Number of patients - Train: {len(train_dataset)}, Dev: {len(val_dataset)}, Test: {len(test_dataset) if config.test else 'N/A'}")

    final_out = None

    if config.train:
        model = create_model_and_optimizer(config)
        model_path_to_load = model.train(train_dataloader, val_dataloader)
    else:
        model_path_to_load = config.model_to_load


    logging.info("Training Complete.")

    if config.val:
        logging.info("Evaluating on validation set using the best model.")
        model = create_model_and_optimizer(config)

        saved_checkpoint = torch.load(
            f=model_path_to_load,
            map_location=
            lambda storage, loc: storage.cuda(config.device.index) if config.device.type == 'cuda' else storage
        )

        # load state dictionaries
        model.network.load_state_dict(state_dict=saved_checkpoint['network_state_dict'])
        model.optimizer.load_state_dict(state_dict=saved_checkpoint['optimizer_state_dict'])

        val_loss, metrics_full = model.evaluate(val_dataloader, plot_metrics=True)

        final_out = plot_metrics(
            metrics=metrics_full,
            endpoints=config.month_endpoints,
            save_dir=config.out_dir
        )
    return final_out

if __name__ == "__main__":
    logging.info("CUDA: %s", torch.cuda.is_available())

    data, config = get_data_and_config_from_cmdline()

    logging.info("Running on device %s", config.device)

    logging.info("Starting run with config %s", config)

    if config.grid_search:
        with open(config.grid_search_params, "r") as f:
            param_grid = json.load(f)
            logging.info("Loaded grid search parameters: %s", param_grid)

        # Run grid search
        grid_search(param_grid, config, data)

    else:
        single_run(config, data)
