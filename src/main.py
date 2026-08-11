"""
EquiEmbryo AI: an ML fairness auditing tool for IVF embryo-ranking models.

Trains two logistic regression models on the same synthetic embryo
dataset - one on labels corrupted by a simulated biased labeling
process, one on the true labels - then audits and compares how fairly
each treats the two demographic groups.

Run with: python main.py
"""

from data_prep import generate_embryo_data, split_data
from model import (
    create_biased_labels,
    get_features_and_labels,
    train_model,
    evaluate_model,
)
from fairness_audit import (
    fairness_report,
    print_fairness_report,
    plot_group_tpr,
    plot_tpr_comparison,
    fairlearn_validation,
)


def main():
    # 1. Generate and split data
    data = generate_embryo_data(n=10000)
    train_df, test_df = split_data(data)

    # 2. Simulate a biased labeling process on the training set
    train_df = create_biased_labels(train_df)

    # 3. Prepare features/labels for both models
    X_train, y_train_biased, y_train_true, X_test, y_test_true = \
        get_features_and_labels(train_df, test_df)
    groups_test = test_df["group"].values

    # 4. Train Model 1 (biased) and Model 3 (ethics-aware)
    model_biased = train_model(X_train, y_train_biased)
    y_pred_biased, _, acc_biased = evaluate_model(model_biased, X_test, y_test_true)

    model_ethics = train_model(X_train, y_train_true)
    y_pred_ethics, _, acc_ethics = evaluate_model(model_ethics, X_test, y_test_true)

    print(f"Model 1 (Biased) accuracy: {acc_biased:.4f}")
    print(f"Model 3 (Ethics-Aware) accuracy: {acc_ethics:.4f}\n")

    # 5. Fairness audit for each model
    report_biased = fairness_report(y_test_true, y_pred_biased, groups_test)
    report_ethics = fairness_report(y_test_true, y_pred_ethics, groups_test)

    print_fairness_report(report_biased, "Model 1 (Biased)")
    print_fairness_report(report_ethics, "Model 3 (Ethics-Aware)")

    # 6. Visualize the comparison
    plot_group_tpr(report_biased, "Model 1 (Biased)")
    plot_tpr_comparison(report_biased, report_ethics)

    # 7. Cross-validate with the Fairlearn library
    fairlearn_validation(y_test_true, y_pred_biased, y_pred_ethics, groups_test)


if __name__ == "__main__":
    main()
