"""
Trains two embryo-ranking models on the same data:

- Model 1 ("biased"): trained on labels that have been deliberately
  corrupted to simulate a biased labeling process against group 1.
- Model 3 ("ethics-aware"): trained on the true, unbiased labels.

Comparing the two is the core of the fairness audit.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

FEATURES = ["morphology", "symmetry", "fragmentation", "group"]


def create_biased_labels(train_df, flip_probability=0.4):
    """Simulate a biased labeling process: for viable embryos in group 1,
    randomly flip ~40% of labels from viable (1) to not viable (0).
    Returns a copy of train_df with a new 'label_biased' column.
    """
    train_df = train_df.copy()
    biased_labels = train_df["true_viable"].values.copy()

    mask = (train_df["true_viable"] == 1) & (train_df["group"] == 1)
    flip = np.random.binomial(1, flip_probability, size=mask.sum())
    biased_labels[mask] = biased_labels[mask] - flip

    train_df["label_biased"] = biased_labels
    return train_df


def get_features_and_labels(train_df, test_df):
    """Extract X/y arrays for both the biased and true label training
    targets, plus the held out test set (always evaluated against true
    labels).
    """
    X_train = train_df[FEATURES].values
    y_train_biased = train_df["label_biased"].values
    y_train_true = train_df["true_viable"].values

    X_test = test_df[FEATURES].values
    y_test_true = test_df["true_viable"].values

    return X_train, y_train_biased, y_train_true, X_test, y_test_true


def train_model(X_train, y_train, max_iter=1000):
    """Train a logistic regression embryo viability classifier."""
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test_true):
    """Return predictions, predicted probabilities, and accuracy for a
    trained model against the true test labels.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test_true, y_pred)
    return y_pred, y_proba, accuracy
