import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import json

def auroc_curve(models_fpr, models_tpr, models_auc, models_names, plot_title, save_dir=None):
    plt.figure(figsize=(10, 8))
    for i in range(len(models_names)):
        curr_fpr, curr_tpr, curr_rocauc, curr_name = models_fpr[i], models_tpr[i], models_auc[i], models_names[i]
        plt.plot(curr_fpr, curr_tpr, lw=2, label=f"{curr_name} (AUROC = {curr_rocauc:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(plot_title, fontsize=20)
    plt.legend(loc="lower right", fontsize=18)
    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / f"{plot_title}.png", dpi=300)
    else:
        plt.show()

    plt.close()

def rr_curve(models_n_risk, models_rr, models_names, plot_title, n_points=None, save_dir=None):
    plt.figure(figsize=(10, 8))

    for i in range(len(models_names)):
        x = np.asarray(models_n_risk[i])
        y = np.asarray(models_rr[i])
        m = np.isfinite(x) & np.isfinite(y)
        if np.any(m):
            curr_x, curr_y = x[m], y[m]

            if n_points is not None and n_points < len(curr_x):
                idx = np.unique(
                    np.minimum(
                        np.round(
                            np.logspace(0, np.log10(len(curr_x) - 1), n_points)
                        ).astype(int),
                        (len(curr_x)-1)
                    )
                )

                curr_x, curr_y = curr_x[idx], curr_y[idx]

            plt.plot(curr_x, curr_y, lw=2, label=models_names[i])

    plt.xscale('log')
    plt.axvline(1000, ls='--', color='k', alpha=0.5)
    plt.xlabel("n at risk (per 1M patients)", fontsize=18)
    plt.ylabel("Relative risk", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(plot_title, fontsize=20)
    plt.legend(fontsize=18)
    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / f"{plot_title}.png", dpi=300)
    else:
        plt.show()
    plt.close()

def analyze_bootstrap(data_dict, endpoint, ci=0.95):

    records = []

    for model_name in data_dict.keys():
        curr_metrics = data_dict[model_name][str(endpoint)]
        for metric_name, values in curr_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            n = len(values)
            sem = stats.sem(values)

            # Confidence interval using t-distribution
            ci_lower, ci_upper = stats.t.interval(ci, df=n-1, loc=mean_val, scale=sem)

            records.append({
                "model": model_name,
                "metric": metric_name,
                "mean": mean_val,
                "std": std_val,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper
            })

    metrics_df = pd.DataFrame(records)

    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generating final plots for paper')

    parser.add_argument('--res-dir', type=str, default=None,
                        help='Directory containing the full results files output from test runs (default: None)')

    parser.add_argument('--default-endpoint', type=int, default=36,
                        help='The default endpoint to plot by (default: 36)')

    parser.add_argument('--bootstrap', action='store_true', default=False,
                        help='Whether or not to run bootstrap analysis (default: False)')

    parser.add_argument('--time-intervals', action='store_true', default=False,
                        help='Whether or not to plot time-intervals (default: False)')


    args = parser.parse_args()

    data_path = Path(args.res_dir)
    all_data = {}

    if args.bootstrap:
        for file in data_path.rglob("*results.json"):
            with open(file, "rb") as f:
                results = json.load(f)
                all_data[str(file).split("/")[2]] = results


        bootstrap_stats = analyze_bootstrap(all_data, args.default_endpoint)
        bootstrap_stats.to_csv("results/bootstrap_test_runs/bootstrap_results_stats.csv", index=False)

    else:
        all_data = {}

        for file in data_path.glob("*.pkl"):
            print("Loading:", file.name)
            with open(file, "rb") as f:
                dat = joblib.load(f)
                all_data[file.stem] = dat

        if args.time_intervals:
            model_name = "Multi-modal Transformer"
            auc_list_all_endpoints = [all_data[model_name][ep]['auc'] for ep in all_data[model_name]]
            fpr_list_all_endpoints = [all_data[model_name][ep]['fpr'] for ep in all_data[model_name]]
            tpr_list_all_endpoints = [all_data[model_name][ep]['tpr'] for ep in all_data[model_name]]
            rr_curve_list_all_endpoints = [all_data[model_name][ep]['rr_curve'] for ep in all_data[model_name]]
            n_risk_list_all_endpoints = [all_data[model_name][ep]['n_at_risk'] for ep in all_data[model_name]]

            endpoints_names = ["3 Months", "6 Months", "12 Months", "36 Months", "60 Months"]

            auroc_curve(fpr_list_all_endpoints, tpr_list_all_endpoints, auc_list_all_endpoints, endpoints_names,
                    "AUROC Diabetes")

            rr_curve(n_risk_list_all_endpoints, rr_curve_list_all_endpoints, endpoints_names,
                 "Relative Risk Diabetes", n_points=10)


        else:

            model_names = ["MLP", "Original Transformer", "Our Transformer", "Multi-modal Transformer"]
            auc_list_all_models = [all_data[m][args.default_endpoint]['auc'] for m in model_names]
            fpr_list_all_models = [all_data[m][args.default_endpoint]['fpr'] for m in model_names]
            tpr_list_all_models = [all_data[m][args.default_endpoint]['tpr'] for m in model_names]
            rr_curve_list_all_models = [all_data[m][args.default_endpoint]['rr_curve'] for m in model_names]
            n_risk_list_all_models = [all_data[m][args.default_endpoint]['n_at_risk'] for m in model_names]

            auroc_curve(fpr_list_all_models, tpr_list_all_models, auc_list_all_models, model_names,
                        "AUROC Diabetes")

            rr_curve(n_risk_list_all_models, rr_curve_list_all_models, model_names,
                     "Relative Risk Diabetes", n_points=10)
