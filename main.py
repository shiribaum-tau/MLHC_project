import json
import pickle
import random

import numpy as np
import torch

import copy

from dataset.dataset import DiseaseProgressionDataset
from pathlib import Path

from consts_and_config import PAD_TOKEN, UNK_TOKEN, Config, DATA_DIR, GROUP_SPLITS

from models.mlp import MLP
from learn.train import train_model

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

print("CUDA:", torch.cuda.is_available())
results_path = "out"

with open(DATA_DIR / f"{dataset_name}.json") as f:
    data = json.load(f)

max_events_length, vocab = get_config_from_data(data)

config = Config(vocab, max_events_length=max_events_length, pad_size=max_events_length,
                time_embed_dim=12, hidden_dim=24, n_trajectories_per_patient_in_test=10,
                min_trajectory_length=3)

train = DiseaseProgressionDataset(data, config, GROUP_SPLITS.TRAIN)
test = DiseaseProgressionDataset(data, config, GROUP_SPLITS.TEST)
val = DiseaseProgressionDataset(data, config, GROUP_SPLITS.VALIDATION)

random.seed(config.random_seed)
np.random.seed(config.random_seed)
print ("Number of patient for -Train:{}, -Dev:{}, -Test:{}".format(
    train.__len__(), val.__len__(), test.__len__()))

model = MLP(config)

print(model)
print("Working threads: ", torch.get_num_threads())
if torch.get_num_threads() < config.num_workers:
    torch.set_num_threads(config.num_workers)
    print("Adding threads count to {}.".format(torch.get_num_threads()))

# # TRAIN
# epoch_stats, model = train_model(train, val, model, config)
# print("Save train/dev results to {}".format(config.results_path))
# print("TRAINING")
# pickle.dump(config.dict(), open(results_path, 'wb'))
# pickle.dump(epoch_stats, open("{}.{}".format(results_path, "epoch_stats"), 'wb'))
# del epoch_stats
# print("Dump results")
