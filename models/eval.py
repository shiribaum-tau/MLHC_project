import pickle
import joblib
import numpy as np
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import logging
import json

logger = logging.getLogger("eval")
logging.getLogger("eval").addHandler(logging.NullHandler())

def _strip_final_num(k: str) -> str:
    """
    Removes the final underscore and number from a string if present.
    """
    i = k.rfind('_')
    return k[:i] if i != -1 and k[i+1:].isdigit() else k

def write_to_tb(tb_writer, key, val, global_step):
    """
    Writes a scalar value to TensorBoard.
    """
    if val is not None:
        tb_writer.add_scalar(tag=key, scalar_value=val, global_step=global_step)

def append_to_dict(origin_dict, to_append):
    """
    Appends values from to_append dict to origin_dict lists.

    Args:
        origin_dict (dict): Original dictionary.
        to_append (dict): Dictionary to append.

    Returns:
        dict: Combined dictionary.
    """
    result = {}
    for key in origin_dict.keys() | to_append.keys():
        original_value = origin_dict.get(key, [])
        to_append_value = to_append.get(key, [])
        result[key] = original_value + to_append_value
    return result

def mean_dict_values(data):
    """
    Computes mean of list values in a dictionary.

    Args:
        data (dict): Input dictionary.

    Returns:
        dict: Dictionary with mean values.
    """
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

def rr_curve(probs, labels, target=1000, population_size=1_000_000):
    """
    Compute RR, precision, recall, and an implied threshold at the *top-k* point
    corresponding to selecting `target` patients per `population_size`.

    Returns:
        tuple: (rr, precision, recall, implied_threshold, n_at_risk_selected)
    """
    N = labels.size
    P = int(labels.sum())
    incidence = labels.mean()

    if N == 0 or P == 0:
        return None, None, None, None, None  # no data or no positives

    # Sort by predicted risk (descending)
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    sorted_labels = labels[order].astype(np.int64)

    # Exact k for the target-per-population
    k_target = int(round(target * N / population_size))
    k_target = max(1, min(N, k_target))  # clip to [1, N]

    # Cumulative TP up to k
    cumsum_pos = np.cumsum(sorted_labels)

    # RR curve
    all_ks = np.arange(1, N+1)
    precision_all_k = cumsum_pos / all_ks
    rr_all_k = precision_all_k / incidence if incidence > 0 else np.full_like(precision_all_k, np.nan, dtype=float)
    n_at_risk = all_ks / N * population_size

    if cumsum_pos[-1] == 0:
        # no positives at all -> nothing to plot
        rr_all_k_masked = np.full_like(rr_all_k, np.nan, dtype=float)
    else:
        mask = cumsum_pos > 0
        rr_all_k_masked = rr_all_k.astype(float).copy()
        rr_all_k_masked[~mask] = np.nan

    plot_data = {'n_at_risk': n_at_risk, 'rr': rr_all_k_masked}

    # Compute RR at k
    TPk = int(cumsum_pos[k_target - 1])
    precision_k = TPk / k_target
    recall_k = TPk / P
    rr_k = (precision_k / incidence) if incidence > 0 else None

    # "Implied threshold" = score of the k-th ranked item
    # (Note: ties at this value may include >k items if you used >= threshold selection)
    thr_k = float(sorted_probs[k_target - 1])

    return float(rr_k), float(precision_k), float(recall_k), thr_k, plot_data


def compute_metrics(config, results, plot_metrics=False):
    """
    Computes evaluation metrics for each endpoint.

    Args:
        config: Configuration object.
        results (dict): Results dictionary.
        plot_metrics (bool): Whether to compute metrics for plotting.

    Returns:
        dict: Metrics for each endpoint.
    """
    logger.info("Starting compute_metrics")
    metrics = {}
    for endpoint_idx, endpoint in enumerate(config.month_endpoints):
        # logger.info(f"Processing endpoint {endpoint} (index {endpoint_idx})")

        # logger.info("Calling get_probs_and_label")
        probs_for_eval, labels_for_eval = get_probs_and_label(results['probs'], results['idx_of_last_y_to_eval'], results["y"], index=endpoint_idx)
        labels_for_eval = np.array(labels_for_eval)
        probs_for_eval = np.array(probs_for_eval)

        # --- AUC ---
        # logger.info("Computing AUC")
        try:
            auc_value = roc_auc_score(labels_for_eval, probs_for_eval)
        except Exception as e:
            logger.warning(f"Failed to compute AUC for endpoint {endpoint}: {e}")
            auc_value = None

        # --- AUPR ---
        # logger.info("Computing AUPR")
        try:
            precisions, recalls, thresholds = precision_recall_curve(labels_for_eval, probs_for_eval, pos_label=1)
            aupr = auc(recalls, precisions)
        except Exception as e:
            logger.warning(f"Failed to compute AUPR for endpoint {endpoint}: {e}")
            precisions, recalls, thresholds = np.array([]), np.array([]), np.array([])
            aupr = None

        # --- Best F1 and Threshold ---
        # logger.info("Computing best F1 and threshold")
        # Compute F1 directly from precision and recall (vectorized)
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
            f1_scores = np.nan_to_num(f1_scores, nan=0.0)  # Replace NaNs from division by zero

        best_idx = np.argmax(f1_scores[:-1])  # Exclude last point where threshold is inf
        best_f1 = float(f1_scores[best_idx])
        best_threshold = float(thresholds[best_idx] if best_idx < thresholds.size else thresholds[-1])
        precision_at_best_f1_threshold = float(precisions[best_idx])
        recall_at_best_f1_threshold = float(recalls[best_idx])

        metrics[f'auc_{endpoint}'] = auc_value
        metrics[f'aupr_{endpoint}'] = aupr
        metrics[f'best_f1_{endpoint}'] = best_f1
        metrics[f'best_f1_threshold_{endpoint}'] = best_threshold
        metrics[f'precision_at_best_f1_threshold_{endpoint}'] = precision_at_best_f1_threshold
        metrics[f'recall_at_best_f1_threshold_{endpoint}'] = recall_at_best_f1_threshold
        rr_k, precision_k, recall_k, thr_k, plot_data = rr_curve(
            probs_for_eval, labels_for_eval, target=1000, population_size=1_000_000
        )
        metrics[f'precision_at_1000_per_million_{endpoint}'] = precision_k
        metrics[f'recall_at_1000_per_million_{endpoint}'] = recall_k
        metrics[f'rr_at_1000_per_million_{endpoint}'] = rr_k
        metrics[f'threshold_at_1000_per_million_{endpoint}'] = thr_k
        # logger.info("Finished computing best F1 and threshold")

        if plot_metrics:

            threshold_to_use = None
            if config.threshold_method == 'f1':
                threshold_to_use = best_threshold
            elif config.threshold_method == 'rr':
                threshold_to_use = thr_k
            elif config.threshold_method == 'const':
                threshold_to_use = config.class_pred_threshold
            else:
                raise ValueError(f"Unknown threshold_method: {config.threshold_method}")
            logger.info(f"Computing ROC curve and confusion matrix. Using {config.threshold_method} ({threshold_to_use}) for thresholding.")

            fpr, tpr, _ = roc_curve(labels_for_eval, probs_for_eval, pos_label=1)
            preds_for_eval = (np.array(probs_for_eval) >= threshold_to_use).astype(int)
            cm = confusion_matrix(labels_for_eval, preds_for_eval)

            metrics[f'tpr_{endpoint}'] = tpr
            metrics[f'fpr_{endpoint}'] = fpr
            metrics[f'precisions_{endpoint}'] = precisions
            metrics[f'recalls_{endpoint}'] = recalls

            metrics[f'confusion_matrix_{endpoint}'] = cm
            # Add confusion matrix components
            tn, fp, fn, tp = cm.ravel()
            metrics[f'cm_tp_{endpoint}'] = tp
            metrics[f'cm_fp_{endpoint}'] = fp
            metrics[f'cm_tn_{endpoint}'] = tn
            metrics[f'cm_fn_{endpoint}'] = fn

            # Add RR curve
            metrics[f'n_at_risk_{endpoint}'] = plot_data['n_at_risk']
            metrics[f'rr_curve_{endpoint}'] = plot_data['rr']

        logger.info(f"Finished endpoint {endpoint}")

    logger.info("compute_metrics finished")
    return metrics


def output_metrics(metrics, endpoints, save_dir=None, full_results=False, bootstrap_bootstrap_skip_large_output=False):
    """
    Plot ROC and PR curves and Confusion matrices for multiple endpoints.

    Parameters
    ----------
    metrics: dict
    endpoints : list of str, optional
        Names for each endpoint. Defaults to ["Endpoint 1", "Endpoint 2", ...].
    save_dir : str, optional
        Directory to save the plots. If None, plots are just shown.
    full_results : bool, optional
        Whether to save full results.
    bootstrap_bootstrap_skip_large_output : bool, optional
        Whether to skip saving large output in bootstrap mode.

    Returns:
        dict or list: Results saved.
    """
    endpoint_names = [f"{i} Months" for i in endpoints]
    logging.info(f"Writing ROC and PR curves to {save_dir}")

    # --- ROC Curve ---
    plt.figure(figsize=(10, 8))
    for endpoint_idx, endpoint in enumerate(endpoints):
        curr_fpr, curr_tpr, curr_rocauc = metrics[f'fpr_{endpoint}'], metrics[f'tpr_{endpoint}'], metrics[f'auc_{endpoint}']
        plt.plot(curr_fpr, curr_tpr, lw=2, label=f"{endpoint_names[endpoint_idx]} (AUC = {curr_rocauc:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.title("ROC Curves", fontsize=18)
    plt.legend(loc="lower right", fontsize=14)
    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / "roc_curves.png", dpi=300)
    else:
        plt.show()

    plt.close()

    # --- Precision-Recall Curve ---
    plt.figure(figsize=(10, 8))

    for endpoint_idx, endpoint in enumerate(endpoints):
        curr_recall, curr_precision, curr_aupr = metrics[f'recalls_{endpoint}'], metrics[f'precisions_{endpoint}'], metrics[f'aupr_{endpoint}']
        plt.plot(curr_recall, curr_precision, lw=2, label=f"{endpoint_names[endpoint_idx]} (AUPR = {curr_aupr:.3f})")

    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.title("Precision-Recall Curves", fontsize=18)
    plt.legend(loc="upper right", fontsize=14)
    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / "pr_curves.png", dpi=300)
    else:
        plt.show()
    plt.close()

    # --- RR Curve ---
    plt.figure(figsize=(10, 8))

    for i, ep in enumerate(endpoints):
        x = np.asarray(metrics[f"n_at_risk_{ep}"])
        y = np.asarray(metrics[f"rr_curve_{ep}"])
        m = np.isfinite(x) & np.isfinite(y)
        if np.any(m):
            plt.plot(x[m], y[m], lw=2, label=endpoint_names[i])
    plt.xscale('log')
    plt.axvline(1000, ls='--', color='k', alpha=0.5)
    plt.xlabel("n at risk (per 1M patients)")
    plt.ylabel("Relative risk")
    plt.legend()
    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / "rr_curves.png", dpi=300)
    else:
        plt.show()
    plt.close()


    # --- Confusion Matrix ---
    for endpoint_idx, endpoint in enumerate(endpoints):
        cm = metrics[f'confusion_matrix_{endpoint}']
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {endpoint_names[endpoint_idx]}")
        # plt.show()

        if save_dir:
            plt.savefig(save_dir / f"cm_{endpoint_names[endpoint_idx]}.png", dpi=300)

        else:
            plt.show()
        plt.close()

    # Output CSV
    if save_dir:
        results_by_endpoint = {e: {"Endpoint": e} for e in endpoints}
        drop_keys = [
            'fpr', 'tpr', 'precisions', 'recalls',
            'confusion_matrix', 'n_at_risk', 'rr_curve'
        ]
        for k, v in metrics.items():
            metric, endpoint = k.rsplit('_', 1)
            if metric in drop_keys and not full_results:
                continue
            results_by_endpoint[int(endpoint)][metric] = v
        if full_results:
            if not bootstrap_bootstrap_skip_large_output:
                with open(save_dir / "full_results.pkl", "wb") as f:
                    joblib.dump(results_by_endpoint, f)
            results_to_save = results_by_endpoint
        else:
            results_to_save = list(results_by_endpoint.values())
            results_df = pd.DataFrame(results_to_save)
            results_df.to_csv(save_dir / "full_results.csv", index=False)

    return results_to_save
