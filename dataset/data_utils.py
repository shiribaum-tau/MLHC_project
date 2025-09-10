from datetime import datetime

import numpy as np
import torch
from consts_and_config import DATE_FORMAT, END_OF_TIME, GROUP_SPLITS, PAD_TOKEN, Config


def concat_collate(batch):
    concat_batch = []
    for sample in batch:
        concat_batch.extend(sample)
    return torch.utils.data.dataloader.default_collate(concat_batch)

def get_dataset_loader(config, data):
    """
    Create a DataLoader for the given dataset.
    Train/Dev/Attribution sets use weighted sampling for balance. Test set uses regular sampling.
    """
    # Determine batch size
    batch_size = config.train_batch_size if data.split_group == GROUP_SPLITS.TRAIN else config.eval_batch_size
    
    # Common DataLoader arguments
    common_args = {
        'batch_size': batch_size,
        'num_workers': config.num_workers,
        'pin_memory': True,
        'collate_fn': concat_collate
    }
    
    # Use weighted sampling for balanced training/validation sets
    if data.split_group != GROUP_SPLITS.TEST:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=data.weights,
            num_samples=len(data),
            replacement=(data.split_group == GROUP_SPLITS.TRAIN)
        )
        return torch.utils.data.DataLoader(data, sampler=sampler, **common_args)
    
    # Use regular sampling for test set
    return torch.utils.data.DataLoader(
        data, 
        shuffle=True, 
        drop_last=False, 
        **common_args
    )


def parse_date(date_str, format=DATE_FORMAT):
    return datetime.strptime(date_str, format)


def pad_arr(arr, max_len, pad_value):
    return np.array([pad_value] * (max_len - len(arr)) + arr[-max_len:])


def is_valid_trajectory(events_to_date, outcome_date, future_cancer, config: Config):
    """
    This function checks whether a single trajectory is valid. A trajectory is valid if:
     (1) It contains enough events.

    And if the patient is a cancer patient,
     (2) The trajectory must end before the pancreatic cancer event.
     (3) The cancer event must occurr within the certain time after the time of assessment.

    Or if the patient is not a cancer patient
     (4) The trajectory must end at least min_followup_for_ctrls_mnths before the end of the dataset
         to exclude those cancer patients died of other reasons with the cancer undetected.

    """
    y = False

    # Filter (1)
    enough_events_counted = len(events_to_date) >= config.min_trajectory_length
    if not enough_events_counted:
        return False, y

    last_event_admission_date = events_to_date[-1]['diag_date']
    months_to_outcome = (outcome_date - last_event_admission_date).days // 30
    last_endpoint = max(config.month_endpoints)

    if future_cancer and (months_to_outcome <= config.exclusion_interval_mnths):
        # Exclusion interval for positives
        # Note: Positive patients where the cancer happened before the last event are filtered here.
        return False, y

    # Filter (2-3)
    is_pos_pre_cancer = last_event_admission_date < outcome_date
    is_pos_in_time_horizon = months_to_outcome < last_endpoint
    is_valid_pos = future_cancer and is_pos_pre_cancer and is_pos_in_time_horizon

    # Filter (4)
    is_valid_neg = (not future_cancer) and (months_to_outcome >= config.min_followup_for_ctrls_mnths)
    
    y = ((months_to_outcome < last_endpoint) and future_cancer)
    return is_valid_neg or is_valid_pos, y


def get_avai_trajectory_indices(patient, events, config: Config):
    """
        This function takes a patients and its events, age and gender information
        and returns all the valid trajectories indexes depending on the filters applied.

        Filters implemented in this version:
            - Remove patients that are not of the given split group
            - Exclusion interval: Removes events too close to PC. If exclusion interval is 0 then do not remove. 

        Returns:
            valid_indices (list of int): Each index is corresponding to one partial trajectory that passed the filter.
            y (bool): If valid_indices is not empty, then y indicates whether any of the trajectories
                      include a cancer diagnosis.

    """
    valid_indices = []
    y = False
    # if config.start_at_attendance:
    #     prefix_end = next((i for i, e in enumerate(events) if e["diag_date"] >= patient['attendance_date']), None)
    #     if prefix_end is None: # attendance_date is after all events
    #         prefix_end = len(events)
    # else:
    #     prefix_end = 0
    prefix_end = 0

    dates = [e['diag_date'] for e in events]
    if config.trajectory_step_by_date:
        # indices of the last occurrence of each consecutive run of the same date
        indices_to_check = [i for i in range(len(dates))
                            if i == len(dates) - 1 or dates[i+1] != dates[i]]
    else:
        indices_to_check = list(range(len(events)))

    # honor prefix_end
    indices_to_check = [i for i in indices_to_check if i >= prefix_end]

    for idx in indices_to_check: # idx is the end of the trajectory
        valid, current_y = is_valid_trajectory(events[:idx+1], patient['outcome_date'], patient['future_cancer'], config)

        if valid:
            valid_indices.append(idx)
            y = current_y or y

    return valid_indices, y


def process_events(events, config: Config):
    """
    Process the diagnosis events depending on the filters. If only known risk factors are used,
    then ICD codes that are not in the subset are replaced with PAD token.
    Also sorts the events.
    """
    for event in events:
        event['diag_date'] = parse_date(event['diagdate'])

    events = sorted(events, key=lambda x: x['diag_date'])

    if config.risk_factor_tokens is not None:
        for e in events:
            if e['codes'] not in config.risk_factor_tokens and e['codes'] != config.target_token:
                e['codes'] = PAD_TOKEN

    return events

def get_outcome_date(events, config: Config, time_of_death=None):
    """
    Looks through events to find date of outcome, which is defined as either pancreatic cancer
    occurrence time or the end of trajectory. If multiple cancer events exist, use the first diagnosis date.

    config:
        events: A list of event dicts. Each dict must have a CODE and diag_date.
        end_of_date: The date for the death for the patient or the end date for
                        the entire dataset (e.g. the patient is still alive).

    Returns:
        ever_develops_cancer (bool): Assess if any given partial trajectory has at least
                                            one diagnosis of pancreatic cancer.
        time (datetime): The Date of pancreatic cancer diagnosis for cases (cancer patients) or
                            END_OF_TIME for controls (normal patients)

    """
    target_events = [e for e in events if config.target_token == e['codes']]

    if len(target_events) > 0:
        ever_develops_cancer = True
        time = min([e['diag_date'] for e in target_events])
    else:
        ever_develops_cancer = False
        time = parse_date(time_of_death) if time_of_death is not None else END_OF_TIME
    return ever_develops_cancer, time
