import numpy as np
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
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


def compute_metrics(config, results, plot_metrics=False):
    metrics = {}
    for endpoint_idx, endpoint in enumerate(config.month_endpoints):

        probs_for_eval, labels_for_eval = get_probs_and_label(results['probs'], results['idx_of_last_y_to_eval'], results["y"], index=endpoint_idx)

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
            precisions = []
            recalls = []
            aupr = None

        metrics[f'auc_{endpoint}'] = auc_value
        metrics[f'aupr_{endpoint}'] = aupr

        if plot_metrics:
            fpr, tpr, _ = roc_curve(labels_for_eval, probs_for_eval, pos_label=1)
            metrics[f'tpr_{endpoint}'] = tpr
            metrics[f'fpr_{endpoint}'] = fpr
            metrics[f'precisions_{endpoint}'] = precisions
            metrics[f'recalls_{endpoint}'] = recalls

    # c_index = compute_c_index(probs, censor_times, golds)

    return metrics


def plot_multi_roc_pr(metrics, endpoints, save_dir=None):
    """
    Plot ROC and PR curves for multiple endpoints.

    Parameters
    ----------
    metrics: dict
    endpoints : list of str, optional
        Names for each endpoint. Defaults to ["Endpoint 1", "Endpoint 2", ...].
    save_dir : str, optional
        Directory to save the plots. If None, plots are just shown.
    """

    endpoint_names = [f"{i} Months" for i in endpoints]
    logging.info(f"Writing ROC and PR curves to {save_dir}")

    # --- ROC Curve ---
    plt.figure(figsize=(7, 6))
    for endpoint_idx, endpoint in enumerate(endpoints):
        curr_fpr, curr_tpr, curr_rocauc = metrics[f'fpr_{endpoint}'], metrics[f'tpr_{endpoint}'], metrics[f'auc_{endpoint}']
        plt.plot(curr_fpr, curr_tpr, lw=2, label=f"{endpoint_names[endpoint_idx]} (AUC = {curr_rocauc:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / "roc_curves.png", dpi=300)
    else:
        plt.show()

    plt.close()

    # --- Precision-Recall Curve ---
    plt.figure(figsize=(7, 6))

    for endpoint_idx, endpoint in enumerate(endpoints):
        curr_recall, curr_precision, curr_aupr = metrics[f'recalls_{endpoint}'], metrics[f'precisions_{endpoint}'], metrics[f'aupr_{endpoint}']
        plt.plot(curr_recall, curr_precision, lw=2, label=f"{endpoint_names[endpoint_idx]} (AUPR = {curr_aupr:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / "pr_curves.png", dpi=300)

        results_to_save = []
        for endpoint in endpoints:
            results_to_save.append({
                "Endpoint": endpoint,
                "AUC": metrics[f'auc_{endpoint}'],
                "AUPR": metrics[f'aupr_{endpoint}']
            })
        results_df = pd.DataFrame(results_to_save)
        results_df.to_csv(save_dir / "auc_aupr_results.csv", index=False)

    else:
        plt.show()
    plt.close()

    return results_to_save
