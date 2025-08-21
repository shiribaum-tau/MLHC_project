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

from consts_and_config import PAD_TOKEN, UNK_TOKEN, Config, DATA_DIR, GROUP_SPLITS

from models.mlp import MLP
from models.model import Model
# from train_model import train_model

dataset_name = "sample_data_formatted"

def get_config_from_data(data, req_split_group: GROUP_SPLITS=GROUP_SPLITS.TRAIN):
    event_lens = []
    vocab = []
    for patient_data in data.values():
        patient_split_group = patient_data.get('split_group', GROUP_SPLITS.TRAIN.value)

        if req_split_group == GROUP_SPLITS.ALL or patient_split_group == req_split_group.value:
            event_lens.append(len(patient_data['events']))
            vocab.extend([event['codes'] for event in patient_data['events']])

    vocab_list = [PAD_TOKEN, UNK_TOKEN] + sorted(list(set(vocab)))
    
    return max(event_lens) if event_lens else 0, {v: k for k, v in enumerate(vocab_list)}

def get_device(device_name):
    if device_name == 'gpu' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    elif device_name == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    return device

print("CUDA:", torch.cuda.is_available())
results_path = "out"

with open(DATA_DIR / f"{dataset_name}.json") as f:
    data = json.load(f)

max_events_length, vocab = get_config_from_data(data)

config = Config(vocab, max_events_length=max_events_length, pad_size=max_events_length,
                time_embed_dim=12, hidden_dim=24, n_trajectories_per_patient_in_test=10,
                min_trajectory_length=3, num_workers=0)

# Device
config.device = get_device(config.device_name)

os.makedirs(config.log_dir, exist_ok=True)
os.makedirs(config.out_dir, exist_ok=True)

# Create Datasets
train = DiseaseProgressionDataset(data, config, GROUP_SPLITS.TRAIN)
test = DiseaseProgressionDataset(data, config, GROUP_SPLITS.TEST)
val = DiseaseProgressionDataset(data, config, GROUP_SPLITS.VALIDATION)
print ("Number of patient for -Train:{}, -Dev:{}, -Test:{}".format(
    train.__len__(), val.__len__(), test.__len__()))

# Set Random Seed
random.seed(config.random_seed)
np.random.seed(config.random_seed)
torch.manual_seed(config.random_seed)

train_dataloader = get_dataset_loader(config, train)
val_dataloader = get_dataset_loader(config, val)
test_dataloader = get_dataset_loader(config, test)

# Create model
network = MLP(config)
network.to(config.device)
optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)

model = Model(network=network, optimizer=optimizer, config=config)

model.train(train_dataloader, val_dataloader)

import ipdb;ipdb.set_trace()
