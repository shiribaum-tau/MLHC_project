
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger("eval")
logging.getLogger("eval").addHandler(logging.NullHandler())

def write_to_tb(tb_writer, key, val, global_step):
    if val is not None:
        tb_writer.add_scalar(tag=key, scalar_value=val, global_step=global_step)

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
            auc_value = roc_auc_score(labels_for_eval, probs_for_eval)
        except Exception as e:
            logger.warning(f"Failed to compute AUC for endpoint {endpoint}: {e}")
            auc_value = None

        try:
            precisions, recalls, _ = precision_recall_curve(labels_for_eval, probs_for_eval, pos_label=1)
            aupr = auc(recalls, precisions)
        except Exception as e:
            logger.warning(f"Failed to compute AUPR for endpoint {endpoint}: {e}")
            aupr = None

        metrics[f'auc_{endpoint}'] = auc_value
        metrics[f'aupr_{endpoint}'] = aupr

    # c_index = compute_c_index(probs, censor_times, golds)

    return metrics

