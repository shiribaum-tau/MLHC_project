from torch.utils import data
from collections import Counter
import numpy as np
import random
import torch

from consts_and_config import Config, GROUP_SPLITS, PAD_TOKEN, UNK_TOKEN
from dataset.data_utils import get_avai_trajectory_indices, get_outcome_date, pad_arr, parse_date, process_events


MAX_TIME_EMBED_PERIOD_IN_DAYS = 120 * 365
MIN_TIME_EMBED_PERIOD_IN_DAYS = 10
SUMMARY_MSG = "Constructed disease progression {} dataset with {} records from {} patients, " \
              "and the following class balance:\n  {}"


class DiseaseProgressionDataset(data.Dataset):
    def __init__(self, data, config: Config, split_group: GROUP_SPLITS):
        """
            Dataset for survival analysis based on categorical disease history information.

        Returns:
            torch.utils.data.Dataset

        """
        super(DiseaseProgressionDataset, self).__init__()
        self.config = config
        self.split_group = split_group
        self.patients = []
        total_positive = 0

        for patient in data:
            patient_data = data[patient]
            patient_dict = {'patient_id': patient}

            if 'split_group' not in patient_data:
                patient_data['split_group'] = GROUP_SPLITS.TRAIN.value
            if split_group != GROUP_SPLITS.ALL and patient_data['split_group'] != split_group.value:
                # print("Skipping patient {} in group {}".format(patient, split_group))
                continue
            if len(patient_data['events']) == 0:
                print("Skipping patient {} with no events".format(patient))
                continue

            events = process_events(patient_data['events'], self.config)
            future_cancer, outcome_date = get_outcome_date(events, self.config, patient_data['death_date'])

            patient_dict.update({'future_cancer': future_cancer,
                                 'dob': parse_date(patient_data['birth_date']),
                                 'attendance_date': parse_date(patient_data['attendance_date']),
                                 'outcome_date': outcome_date,
                                 'split_group': patient_data['split_group']})

            avai_indices, any_y = get_avai_trajectory_indices(patient_dict, events, self.config)
            if any_y:
                total_positive += 1
            patient_dict.update({'avai_indices': avai_indices, 'y': any_y, 'events': events})

            if avai_indices:
                self.patients.append(patient_dict)

        print("Number of positive patients  in '{}' dataset is: {}.".format(self.split_group.value, total_positive))
        self.calculate_class_weights()

    def calculate_class_weights(self):
        """
            Calculates the weights used by WeightedRandomSampler for balancing the batches. 
        """
        ys = [patient['y'] for patient in self.patients]
        label_counts = Counter(ys)
        weight_per_label = 1. / len(label_counts)
        label_weights = {
            label: weight_per_label/count for label, count in label_counts.items()
        }
        self.weights = [label_weights[d] for d in ys]


    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):
        if isinstance(index, slice):
            # Handle slice objects
            indices = range(*index.indices(len(self.patients)))
            return [self._get_single_item(i) for i in indices]
        else:
            # Handle single integer index
            return self._get_single_item(index)
    
    def _get_single_item(self, index):
        patient = self.patients[index]
        samples = self.get_trajectory(patient)
        items = []
        for sample in samples:
            code_str = " ".join(sample['codes'])
            x = [self.get_index_for_code(code) for code in sample['codes']]
            time_seq = sample['time_seq'].tolist()
            age_seq = sample['age_seq'].tolist()
            item = {
                'x': pad_arr(x, self.config.pad_size, 0),
                'time_seq': pad_arr(time_seq, self.config.pad_size, np.zeros(self.config.time_embed_dim)),
                'age_seq': pad_arr(age_seq, self.config.pad_size, np.zeros(self.config.time_embed_dim)),
                # 'code_str': code_str
            }
            for key in ['y', 'y_seq', 'y_mask', 'idx_of_last_y_to_eval', 'age', 'future_cancer',
                        'days_to_censor', 'patient_id']: #'diag_date', , 'patient_id'
                item[key] = sample[key]
            items.append(item)
        return items

    def get_index_for_code(self, code):
        return self.config.vocab.get(code, self.config.vocab[UNK_TOKEN])

    def get_trajectory(self, patient):
        """
            Given a patient, multiple trajectories can be extracted by sampling partial histories.
        """
        if self.split_group != GROUP_SPLITS.TRAIN:
            selected_idx = [random.choice(patient['avai_indices']) for _ in range(self.config.n_trajectories_per_patient_in_test)]
        else:
            selected_idx = [random.choice(patient['avai_indices'])]

        samples = []
        for idx in selected_idx:
            events_to_date = patient['events'][:idx + 1]

            codes = [e['codes'] for e in events_to_date]
            _, time_seq = self.get_time_seq(events_to_date, events_to_date[-1]['diag_date'])
            age, age_seq = self.get_time_seq(events_to_date, patient['dob'])
            y, y_seq, y_mask, idx_of_last_y_to_eval, days_to_censor = self.get_label(patient, until_idx=idx)
            samples.append({
                'codes': codes,
                'y': y,
                'y_seq': y_seq,
                'y_mask': y_mask,
                'idx_of_last_y_to_eval': idx_of_last_y_to_eval,
                'future_cancer': patient['future_cancer'],
                'patient_id': patient['patient_id'],
                'days_to_censor': days_to_censor,
                'time_seq': time_seq,
                'age_seq': age_seq,
                'age': age
                # 'diag_date': events_to_date[-1]['diag_date'].isoformat()
            })
        return samples

    def get_time_seq(self, events, reference_date):
        """
            Calculates the positional embeddings depending on the time diff from the events and the reference date.
        """
        deltas = np.array([abs((reference_date - event['diag_date']).days) for event in events])
        multipliers = 2*np.pi / (np.linspace(
            start=MIN_TIME_EMBED_PERIOD_IN_DAYS, stop=MAX_TIME_EMBED_PERIOD_IN_DAYS, num=self.config.time_embed_dim
        ))
        deltas, multipliers = deltas.reshape(len(deltas), 1), multipliers.reshape(1, len(multipliers))
        positional_embeddings = np.cos(deltas*multipliers)
        return max(deltas), positional_embeddings

    def get_label(self, patient, until_idx):
        """
        config:
            patient (dict): The patient dictionary which includes all the processed diagnosis events.
            until_idx (int): Specify the end point for the partial trajectory.

        Returns:
            outcome_date: date of pancreatic cancer diagnosis for cases (cancer patients) or
                          END_OF_TIME_DATE for controls (normal patients)
            idx_of_last_y_to_eval: the position in time vector (default: [3,6,12,36,60]) which specify the outcome_date
            y_seq: Used as golds in cumulative_probability_layer
                   An all zero array unless ever_develops_panc_cancer then y_seq[idx_of_last_y_to_eval:]=1
            y_mask: how many years left in the disease window
                    ([1] for 0:idx_of_last_y_to_eval years and [0] for the rest)
                    (without linear interpolation, y_mask looks like complement of y_seq)

            Ex1:  A partial disease trajectory that includes pancreatic cancer diagnosis between 6-12 months
                  after time of assessment.
                    idx_of_last_y_to_eval: 2
                    y_seq: [0, 0, 1, 1, 1]
                    y_mask: [1, 1, 1, 0, 0]
            Ex2:  A partial disease trajectory from a patient who never gets pancreatic cancer diagnosis
                  but died between 36-60 months after time of assessment.
                    idx_of_last_y_to_eval: 1
                    y_seq: [0, 0, 0, 0, 0]
                    y_mask: [1, 1, 1, 1, 0]
        """
        last_event = patient['events'][until_idx]
        days_to_censor = (patient['outcome_date'] - last_event['diag_date']).days
        num_time_steps, max_time = len(self.config.month_endpoints), max(self.config.month_endpoints)
        # Does the patient become positive within the timeframe
        y = days_to_censor < (max_time * 30) and patient['future_cancer']
        y_seq = np.zeros(num_time_steps)
        if days_to_censor < (max_time * 30):
            # if cancer within timeframe, find the first y index that should be 1
            idx_of_last_y_to_evaluate = min([i for i, mo in enumerate(self.config.month_endpoints) if days_to_censor < (mo*30)])
        else:
            idx_of_last_y_to_evaluate = num_time_steps - 1

        if y:
            y_seq[idx_of_last_y_to_evaluate:] = 1
        y_mask = np.array([1] * (idx_of_last_y_to_evaluate+1) + [0] * (num_time_steps - (idx_of_last_y_to_evaluate+1)))

        assert idx_of_last_y_to_evaluate >= 0 and len(y_seq) == len(y_mask)
        return y, y_seq.astype('float64'), y_mask.astype('float64'), idx_of_last_y_to_evaluate, days_to_censor

