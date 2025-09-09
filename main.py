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
from models.eval import output_metrics

from models.mlp import MLP
from models.mm_transformer import MMTransformer
from models.model import Model
from models.transformer import Transformer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("main")
logging.getLogger("main").addHandler(logging.NullHandler())

def make_unique_dir(base_dir):
    base_dir = str(base_dir)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return base_dir
    i = 1
    while True:
        new_dir = f"{base_dir}_{i}"
        if not os.path.exists(new_dir):
            logger.info(f"Directory {base_dir} already exists, using {new_dir} instead.")
            os.makedirs(new_dir)
            return new_dir
        i += 1

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
        logger.info(f"Grid search results saved to {csv_path}")
    

def single_run(config, data):
    # Set Random Seed
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    # Make run directories, appending a number if they already exist
    make_unique_dir(config.log_dir)
    make_unique_dir(config.out_dir)

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

    logger.info(f"Number of patients - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset) if config.test else 'N/A'}")

    final_out = None

    if config.train:
        model = create_model_and_optimizer(config)
        model_path_to_load = model.train(train_dataloader, val_dataloader)
    else:
        model_path_to_load = config.model_to_load


    logger.info("Training Complete.")

    if config.val:
        logger.info("Evaluating on validation set using the best model.")
        model = create_model_and_optimizer(config)

        saved_checkpoint = torch.load(
            f=model_path_to_load,
            map_location=
            lambda storage, loc: storage.cuda(config.device.index) if config.device.type == 'cuda' else storage
        )

        # load state dictionaries
        model.network.load_state_dict(state_dict=saved_checkpoint['network_state_dict'])
        model.optimizer.load_state_dict(state_dict=saved_checkpoint['optimizer_state_dict'])

        _, metrics_full = model.evaluate(val_dataloader, plot_metrics=True)

        final_out = output_metrics(
            metrics=metrics_full,
            endpoints=config.month_endpoints,
            save_dir=config.out_dir
        )
    return final_out

def replace_config_for_val(config: Config) -> Config:
    excluded_keys = ['run_name', 'run_dir', 'train', 'val', 'test', 'grid_search']
    path_to_config = config.model_to_load_dir / "config.json"
    with open(path_to_config, "r") as f:
        config_dict = json.load(f)
    # Only keep keys in config_dict that are fields of Config
    config_fields = set(config.__dataclass_fields__.keys())
    filtered_config_dict = {k: v for k, v in config_dict.items() if k in config_fields and k not in excluded_keys}
    alt = [k for k in config_dict.keys() if k not in config_fields]
    logger.info(f"Keys in loaded config not in Config dataclass and will be ignored: {alt}")
    config = replace(config, **filtered_config_dict)
    return config

def diff_configs(base_config, new_config, exclude=['val', 'train', 'test', 'grid_search',
                                                   'run_name', 'run_dir', 'device_name', 'start_time',
                                                   'model_to_load_dir', 'grid_search_params', 'num_epochs', 'month_endpoints']):
    base_d, new_d = base_config.dict(), new_config.dict()
    return {
        k: new_d.get(k)
        for k in base_d.keys()
        if base_d.get(k) != new_d.get(k) and k not in exclude
    }

def nunique_safe(s: pd.Series) -> int:
    return s.map(lambda x: tuple(x) if isinstance(x, list) else x).nunique(dropna=False)

def bulk_val(base_config: Config, data):
    os.makedirs(base_config.run_dir, exist_ok=True)
    # with open(base_config.run_dir / "base_config.json", "w") as f:
    #     json.dump(base_config.dict(), f)

    results = []
    csv_path = base_config.run_dir / "bulk_val_results.csv"
    same_values_path = base_config.run_dir / "same_values.json"
    for root, dirs, files in os.walk(base_config.result_dir_for_val):
        if "config.json" not in files or base_config.model_to_load_name not in files:
            continue
        logger.info(f"Processing directory: {root}")
        # Create new config
        config = replace(base_config, model_to_load_dir=root, val=True, train=False)
        config = replace_config_for_val(config)
        diffs = diff_configs(base_config, config)
        new_run_name = f"{base_config.run_name}_bulkval_{'__'.join(f'{k}_{val}' for k, val in diffs.items())}"
        new_run_dir = base_config.run_dir / new_run_name
        config = replace(config, run_dir=new_run_dir, run_name=new_run_name)
        out = single_run(config, data)
        for result in out:
            results.append({**config.dict(), **result})

        df = pd.DataFrame(results)
        const_cols = {col: df[col].iloc[0] for col in df.columns if nunique_safe(df[col]) == 1}
        df = df.drop(columns=list(const_cols.keys()) + ['run_name','run_dir'], errors='ignore')
        df.to_csv(csv_path, index=False)
        with open(same_values_path, "w") as f:
            json.dump(const_cols, f, indent=4, default=str)
        logger.info(f"Bulk validation results saved to {csv_path}")


if __name__ == "__main__":
    logger.info("CUDA: %s", torch.cuda.is_available())

    data, config = get_data_and_config_from_cmdline()

    logger.info("Running on device %s", config.device)

    logger.info("Starting run with config %s", config)

    if config.grid_search:
        with open(config.grid_search_params, "r") as f:
            param_grid = json.load(f)
            logger.info("Loaded grid search parameters: %s", param_grid)

        # Run grid search
        grid_search(param_grid, config, data)
    elif config.bulk_val:
        bulk_val(config, data)
    else:
        if config.val and not config.train:
            config = replace_config_for_val(config)

        single_run(config, data)
