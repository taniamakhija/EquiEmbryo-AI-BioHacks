# EquiEmbryo-AI-biohacks
Goal: Reducing Bias in IVF Embryo Ranking

EquiEmbryo AI explores fairness in AI systems used to rank IVF embryos. An ML fairness auditing tool for IVF embryo-ranking models, built for BioHacks 2026 (1st Place, BioEthics Track).

AI models trained on historical data can inherit hidden bias. If some demographic groups had unequal access to care in the pass the model may treat embryos from those groups unfairly. We simulate this problem and show how to fix it.

## Links

- [Devpost submission](https://devpost.com/software/equiembryo-ai)

## What it does

Trains two logistic regression models on the same synthetic embryo dataset:

- **Model 1 (Biased):** trained on labels corrupted by a simulated biased labeling process that under-labels viable embryos in one demographic group.
- **Model 3 (Ethics-Aware):** trained on the true, unbiased labels.

It then audits both models for fairness by comparing true positive rate (TPR) and false positive rate (FPR) across demographic groups, and cross-validates the results using the [Fairlearn](https://fairlearn.org/) library.

## Key result

The biased model shows a large TPR gap between groups (~0.34), meaning it systematically under-predicts viability for one group. The ethics-aware model, trained on unbiased labels, shrinks that gap to ~0.04.

## Structure
