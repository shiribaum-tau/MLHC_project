from dataclasses import replace
import json
import logging
from consts_and_config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("conf_utils")
logging.getLogger("conf_utils").addHandler(logging.NullHandler())


def replace_config_for_val(config: Config) -> Config:
    """
    Loads config from file and replaces fields for validation.

    Args:
        config (Config): Base config.

    Returns:
        Config: Updated config for validation.
    """
    excluded_keys = ['dataset_name', 'device_name', 'run_name', 'run_dir', 'bootstrap_test',
                     'bootstrap_repeats', 'subsample_ratio',
                     'train', 'val', 'test', 'grid_search', 'bulk_val', 'data_dir',
                     'grid_search_params', 'result_dir_for_val', 'model_to_load_dir',
                     'model_to_load_name', 'random_seed', 'start_time']
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
    """
    Computes the difference between two configs, excluding specified keys.

    Args:
        base_config (Config): Base config.
        new_config (Config): New config.
        exclude (list): Keys to exclude from comparison.

    Returns:
        dict: Dictionary of differing keys and their new values.
    """
    base_d, new_d = base_config.dict(), new_config.dict()
    return {
        k: new_d.get(k)
        for k in base_d.keys()
        if base_d.get(k) != new_d.get(k) and k not in exclude
    }

