import datetime
import os
from enum import Enum, auto
import pathlib
from typing import Union
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
DATE_FORMAT = "%Y-%m-%d"
END_OF_TIME = datetime.datetime.max

ROOT_DIR = Path(".")


class GROUP_SPLITS(Enum):
    TRAIN="train"
    VALIDATION="val"
    TEST="test"
    ALL="all"

@dataclass()
class Config:
    # Required fields (no defaults) - grouped by category
    
    # Core required fields
    vocab: dict
    dataset_name: str
    data_dir: pathlib.Path

    # Device and basic configuration
    run_name: str
    device_name: str
    device: torch.device
    random_seed: int
    out_dir: pathlib.Path
    log_dir: pathlib.Path
    start_time: str

    # Data preprocessing configuration
    start_at_attendance: bool
    min_trajectory_length: int
    min_followup_for_ctrls_mnths: int
    exclusion_interval_mnths: int
    max_events_length: int
    pad_size: int

    # Model architecture
    hidden_dim: int
    time_embed_dim: int
    num_layers: int
    dropout: float
    pool_name: str

    # Training configuration
    learning_rate: float
    num_epochs: int
    train_batch_size: int
    eval_batch_size: int
    num_workers: int
    resume_epoch: int

    # Evaluation configuration
    n_trajectories_per_patient_in_test: int
    n_batches: int
    n_train_batches_per_eval: int
    n_batches_for_eval: int

    # Fields with defaults
    target_tokens: tuple = ("642",) # ("C25",)
    risk_factor_tokens: tuple = None
    month_endpoints: tuple = (3, 6, 12, 36, 60)


    def dict(self):
        excluded_keys = ['vocab', 'device']
        full_dict = asdict(self)
        ret = {k: v for k, v in full_dict.items() if k not in excluded_keys}

        for path_key in ['data_dir', 'log_dir', 'out_dir']:
            ret[path_key] = str(ret[path_key])

        return ret

    def __repr__(self):
        config_dict = self.dict()
        params = ', '.join(f'{k}={repr(v)}' for k, v in config_dict.items())
        return f"Config({params})"

    def __str__(self):
        config_dict = self.dict()
        params = '\n  '.join(f'{k}: {v}' for k, v in config_dict.items())
        return f"Config:\n  {params}"
