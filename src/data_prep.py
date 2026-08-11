"""
Generates a synthetic embryo dataset and splits it into train/test sets.
The dataset is built to be fair by design: viability does not depend on
group membership. Bias is only introduced later, deliberately, in
model.py, to simulate what a biased labeling process would look like.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)


def generate_embryo_data(n=10000):
    """Create a synthetic dataset of n embryos with morphology, symmetry,
    fragmentation features, a demographic group label, and a true
    (unbiased) viability label.
    """
    morphology = np.random.rand(n)      # quality of embryo structure, higher is better
    symmetry = np.random.rand(n)        # how symmetrical the embryo is, higher is better
    fragmentation = np.random.rand(n)   # amount of fragmentation, higher is worse

    group = np.random.binomial(1, 0.5, n)  # demographic group, 0 or 1

    # true viability score does not use group, keeping ground truth fair
    noise = np.random.normal(0, 0.1, n)
    score_true = 0.6 * morphology + 0.3 * symmetry - 0.5 * fragmentation + noise

    threshold = 0.3
    true_viable = (score_true > threshold).astype(int)

    data = pd.DataFrame({
        "morphology": morphology,
        "symmetry": symmetry,
        "fragmentation": fragmentation,
        "group": group,
        "true_viable": true_viable,
    })

    return data


def split_data(data, test_size=0.3, random_state=42):
    """Stratified train/test split, preserving the true_viable ratio in
    both sets.
    """
    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data["true_viable"],
    )
    return train_df.copy(), test_df.copy()
