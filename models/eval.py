
import warnings
import numpy as np
import sklearn
from dataclasses import dataclass, asdict


def append_to_dict(origin_dict, to_append):
    result = {}
    for key in origin_dict.keys() | to_append.keys():
        original_value = origin_dict.get(key, [])
        to_append_value = to_append.get(key, [])
        result[key] = original_value + to_append_value
    return result

def mean_dict_values(data):
    return {
        k: np.mean(v) if isinstance(v, list) else v
        for k, v in data.items()
    }

def get_probs_and_label(probs, idx_of_last_y_to_eval, y, index=4):
    probs_for_eval, labels_for_eval = [], []
    for prob_arr, first_positive_y, y_true in zip(probs, idx_of_last_y_to_eval, y):
        valid_pos = y_true and first_positive_y <= index
        valid_neg = first_positive_y >= index
        label = valid_pos
        if valid_pos or valid_neg:
            probs_for_eval.append(prob_arr[index])
            labels_for_eval.append(label)

    return probs_for_eval, labels_for_eval


def compute_metrics(config, results):
    metrics = {}
    for endpoint_idx, endpoint in enumerate(config.month_endpoints):

        probs_for_eval, labels_for_eval = get_probs_and_label(results['probs'], results['idx_of_last_y_to_eval'], results["y"], index=endpoint_idx)

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

