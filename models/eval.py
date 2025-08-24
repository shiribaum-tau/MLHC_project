
import warnings
import numpy as np
import sklearn
from dataclasses import dataclass, asdict


def merge_duplicate_keys_flat(dict1, dict2):
    result = {}
    for key in dict1.keys() | dict2.keys():
        values = [
            *([dict1[key]] if key in dict1 and not isinstance(dict1[key], list) else dict1.get(key, [])),
            *([dict2[key]] if key in dict2 and not isinstance(dict2[key], list) else dict2.get(key, []))
        ]
        result[key] = values if len(values) > 1 else values[0]
    return result

def mean_dict_values(data):
    return {
        k: np.mean(v) if isinstance(v, list) else v
        for k, v in data.items()
    }

def get_probs_and_label(probs, idx_of_last_y_to_eval, y, batch, index=4):
    probs_for_eval, labels_for_eval = [], []
    for patient_index, (prob_arr, first_positive_y, y_true) in enumerate(zip(probs, idx_of_last_y_to_eval, y)):

        # Check if there are any positive patients with y_seq=0
        if y_true and not any(batch['y_seq'][patient_index]):
            with open('debug.txt', 'a') as f:
                f.write(f"Patient ID: {batch['patient_id'][patient_index]}, Y True: {y_true}, Seq: {batch['y_seq'][patient_index].numpy()} Trajectory: {batch['x'][patient_index].numpy()}\n")

        valid_pos = y_true and first_positive_y <= index
        valid_neg = first_positive_y >= index
        label = valid_pos
        if valid_pos or valid_neg:
            probs_for_eval.append(prob_arr[index])
            labels_for_eval.append(label)

    return probs_for_eval, labels_for_eval


def compute_metrics(config, probs, batch):
    metrics = {}
    for endpoint_idx, endpoint in enumerate(config.month_endpoints):

        probs_for_eval, labels_for_eval = get_probs_and_label(probs, batch['idx_of_last_y_to_eval'], batch["y"], batch, index=endpoint_idx)

        # fpr, tpr, _ = sklearn.metrics.roc_curve(labels_for_eval, probs_for_eval, pos_label=1)
        # import ipdb;ipdb.set_trace()
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                auc = sklearn.metrics.roc_auc_score(labels_for_eval, probs_for_eval)
                if w:
                    import ipdb; ipdb.set_trace()
        except Exception:
            auc = None

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                precisions, recalls, _ = sklearn.metrics.precision_recall_curve(labels_for_eval, probs_for_eval, pos_label=1)
                aupr = sklearn.metrics.auc(recalls, precisions)
                if w:
                    import ipdb; ipdb.set_trace()
        except Exception:
            aupr = None

        metrics[f'auc_{endpoint}'] = auc
        metrics[f'aupr_{endpoint}'] = aupr

    # c_index = compute_c_index(probs, censor_times, golds)

    return metrics

