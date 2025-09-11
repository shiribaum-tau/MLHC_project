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

CATEGORICAL_TYPES = ["ICD10"]

ROOT_DIR = Path(".")

class GROUP_SPLITS(Enum):
    TRAIN="train"
    VALIDATION="val"
    TEST="test"
    ALL="all"

class SUPPORTED_MODELS(Enum):
    MLP = "mlp"
    TRANSFORMER = "transformer"
    MM_TRANSFORMER = "mm_transformer"

@dataclass()
class Config:
    # Required fields (no defaults) - grouped by category
    
    # Core required fields
    vocab: dict
    token_types: dict
    dataset_name: str
    data_dir: pathlib.Path

    # Main steps to execute
    train: bool
    val: bool
    test: bool
    grid_search: bool
    bulk_val: bool

    # Device and basic configuration
    run_name: str
    device_name: str
    device: torch.device
    random_seed: int
    base_output_dir: pathlib.Path
    result_dir_for_val: pathlib.Path
    run_dir: pathlib.Path
    start_time: str
    model_to_load_dir: pathlib.Path
    model_to_load_name: str
    grid_search_params: pathlib.Path

    # Data preprocessing configuration
    min_trajectory_length: int
    min_followup_for_ctrls_mnths: int
    exclusion_interval_mnths: int
    max_events_length: int
    pad_size: int

    # Model architecture
    model_type: SUPPORTED_MODELS
    hidden_dim: int
    time_embed_dim: int
    num_layers: int
    num_heads: int # For transformer
    dropout: float
    pool_name: str

    # Embedding configuration
    use_time_embed: bool
    use_age_embed: bool

    # Training configuration
    learning_rate: float
    num_epochs: int
    train_batch_size: int
    eval_batch_size: int
    num_workers: int
    resume_epoch: int
    weight_decay: float

    # Evaluation configuration
    n_trajectories_per_patient_in_test: int
    n_batches: int
    n_train_batches_per_eval: int
    n_batches_for_eval: int

    target_token: str # ("C25",)

    # Fields with defaults
    start_at_attendance: bool = False # Unused
    trajectory_step_by_date: bool = False
    risk_factor_tokens: tuple = None
    month_endpoints: tuple = (3, 6, 12, 36, 60)
    min_delta_checkpoint: float = 0.0
    min_delta_earlystopping: float = 0.0
    reduce_lr_patience: int = 5
    lr_decay: float = 0.5
    threshold_method: str = 'f1'  # Options: 'f1', 'rr', 'const'
    class_pred_threshold: float = 0.5 # Used if threshold_method == 'const'

    # log_dir and out_dir are now properties
    @property
    def log_dir(self):
        return self.run_dir / "log"

    @property
    def out_dir(self):
        return self.run_dir / "out"

    @property
    def model_to_load(self):
        if self.model_to_load_dir:
            return self.model_to_load_dir / self.model_to_load_name
        return None

    def __post_init__(self):
        # Convert string arguments to Path where type annotation is pathlib.Path
        for field in self.__dataclass_fields__.values():
            if field.type is pathlib.Path:
                val = getattr(self, field.name)
                if val is not None and not isinstance(val, pathlib.Path):
                    setattr(self, field.name, pathlib.Path(val))
        # Convert string to SUPPORTED_MODELS for model_type
        if hasattr(self, 'model_type') and not isinstance(self.model_type, SUPPORTED_MODELS):
            try:
                self.model_type = SUPPORTED_MODELS(self.model_type)
            except ValueError:
                raise ValueError(f"Invalid model_type '{self.model_type}'. Allowed values: {[m.value for m in SUPPORTED_MODELS]}")

    def dict(self):
        excluded_keys = ['vocab', 'device', 'token_types']
        full_dict = asdict(self)
        ret = {k: v for k, v in full_dict.items() if k not in excluded_keys}

        for k, v in ret.items():
            if isinstance(v, pathlib.Path):
                ret[k] = str(v)
            elif isinstance(v, Enum):
                ret[k] = v.value

        return ret

    def __repr__(self):
        config_dict = self.dict()
        params = ', '.join(f'{k}={repr(v)}' for k, v in config_dict.items())
        return f"Config({params})"

    def __str__(self):
        config_dict = self.dict()
        params = '\n  '.join(f'{k}: {v}' for k, v in config_dict.items())
        return f"Config:\n  {params}"
