"""
Fairness auditing tools: computes per-group true/false positive rates,
builds a comparison report between groups, and plots the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def group_stats(y_true, y_pred, groups, group_value):
    """Compute TPR/FPR for a single demographic group."""
    mask = (groups == group_value)
    y_t = y_true[mask]
    y_p = y_pred[mask]

    tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # of viable embryos, how many correctly predicted
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # of non-viable embryos, how many incorrectly predicted

    return {"n": mask.sum(), "TPR": tpr, "FPR": fpr}


def fairness_report(y_true, y_pred, groups):
    """Compare fairness across group 0 and group 1, including the TPR/FPR
    gaps between them.
    """
    stats0 = group_stats(y_true, y_pred, groups, 0)
    stats1 = group_stats(y_true, y_pred, groups, 1)

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "group_0": stats0,
        "group_1": stats1,
        "TPR_gap": stats0["TPR"] - stats1["TPR"],
        "FPR_gap": stats0["FPR"] - stats1["FPR"],
    }


def print_fairness_report(report, model_name):
    """Print a fairness report in a human-readable format."""
    print(f"{model_name} Fairness Report")
    print("Accuracy:", report["accuracy"])
    print("Group 0 TPR:", report["group_0"]["TPR"])
    print("Group 1 TPR:", report["group_1"]["TPR"])
    print("TPR gap:", report["TPR_gap"])
    print("Group 0 FPR:", report["group_0"]["FPR"])
    print("Group 1 FPR:", report["group_1"]["FPR"])
    print("FPR gap:", report["FPR_gap"])
    print()


def plot_group_tpr(report, model_name):
    """Bar chart of TPR by group for a single model."""
    labels = ["Group 0", "Group 1"]
    tprs = [report["group_0"]["TPR"], report["group_1"]["TPR"]]

    plt.bar(labels, tprs)
    plt.ylim(0, 1)
    plt.ylabel("TPR (Sensitivity)")
    plt.title(f"{model_name} - TPR by Group")
    plt.show()


def plot_tpr_comparison(report_biased, report_ethics):
    """Side-by-side bar chart comparing TPR by group between the biased
    and ethics-aware models.
    """
    labels = ["Group 0", "Group 1"]
    tprs_biased = [report_biased["group_0"]["TPR"], report_biased["group_1"]["TPR"]]
    tprs_ethics = [report_ethics["group_0"]["TPR"], report_ethics["group_1"]["TPR"]]

    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width / 2, tprs_biased, width, label="Model 1 (Biased)")
    plt.bar(x + width / 2, tprs_ethics, width, label="Model 3 (Ethics-Aware)")

    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("TPR (Sensitivity)")
    plt.title("TPR by Group: Biased vs Ethics-Aware")
    plt.legend()
    plt.show()


def fairlearn_validation(y_test_true, y_pred_biased, y_pred_ethics, groups_test):
    """Cross-check the manual fairness report against the Fairlearn
    library's TPR-by-group computation.
    """
    from fairlearn.metrics import MetricFrame, true_positive_rate

    frame_biased = MetricFrame(
        metrics=true_positive_rate,
        y_true=y_test_true,
        y_pred=y_pred_biased,
        sensitive_features=groups_test,
    )
    print("Fairlearn TPR by group for Model 1 (Biased):")
    print(frame_biased.by_group)

    frame_ethics = MetricFrame(
        metrics=true_positive_rate,
        y_true=y_test_true,
        y_pred=y_pred_ethics,
        sensitive_features=groups_test,
    )
    print("\nFairlearn TPR by group for Model 3 (Ethics-Aware):")
    print(frame_ethics.by_group)
