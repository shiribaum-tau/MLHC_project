import datetime
import os
from enum import Enum, auto
from typing import Union
from dataclasses import dataclass, asdict
from pathlib import Path

UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
DATE_FORMAT = "%Y-%m-%d"
END_OF_TIME = datetime.datetime.max

ROOT_DIR = Path(".")

DATA_DIR = ROOT_DIR / "data"

class GROUP_SPLITS(Enum):
    TRAIN="train"
    VALIDATION="dev"
    TEST="test"
    ALL="all"

@dataclass(frozen=True)
class Config:
    vocab: dict
    random_seed: int = 42
    start_at_attendance: bool = True # Always include the attendance date in the trajectory

    target_tokens: tuple = ("C25",)
    risk_factor_tokens: tuple = None
    min_trajectory_length: int = 5
    min_followup_for_ctrls_mnths: int = 24
    exclusion_interval_mnths: int = 0
    month_endpoints: tuple = (3, 6, 12, 36)

    time_embed_dim: int = 128
    max_events_length: int = None
    pad_size: int = None

    n_trajectories_per_patient_in_test: int = 250
    hidden_dim : int = 256
    dropout: float = 0.2
    pool_name: str = 'GlobalAvgPool'
    num_layers: int = 1
    num_workers: int = 8 # for data loader


    def dict(self):
        excluded_keys = ['vocab']
        full_dict = asdict(self)
        ret = {k: v for k, v in full_dict.items() if k not in excluded_keys}
        return ret

    def __repr__(self):
        config_dict = self.dict()
        params = ', '.join(f'{k}={repr(v)}' for k, v in config_dict.items())
        return f"Config({params})"

    def __str__(self):
        config_dict = self.dict()
        params = '\n  '.join(f'{k}: {v}' for k, v in config_dict.items())
        return f"Config:\n  {params}"
