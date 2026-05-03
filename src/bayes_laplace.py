from __future__ import annotations

import numpy as np
from scipy.special import expit


def binary_entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


class DiagonalLaplaceLogistic:
    def __init__(self, coef, intercept, prior_precision: float = 1.0):
        self.coef = np.asarray(coef, dtype=np.float32).reshape(-1)
        self.intercept = float(intercept)
        self.prior_precision = float(prior_precision)
        self.var = None
        self.bias_var = None

    def fit_posterior(self, x_train: np.ndarray) -> "DiagonalLaplaceLogistic":
        logits = x_train @ self.coef + self.intercept
        p = expit(logits)
        weights = p * (1.0 - p)
        # Diagonal Laplace keeps only per-feature curvature, which is cheap enough for many probes.
        precision = self.prior_precision + (weights[:, None] * (x_train ** 2)).sum(axis=0)
        self.var = (1.0 / np.maximum(precision, 1e-8)).astype(np.float32)
        self.bias_var = float(1.0 / max(self.prior_precision + weights.sum(), 1e-8))
        return self

    def predict(self, x: np.ndarray, num_samples: int = 64, seed: int = 1337) -> dict[str, np.ndarray]:
        if self.var is None or self.bias_var is None:
            raise RuntimeError("fit_posterior must be called before predict")
        rng = np.random.default_rng(seed)
        weight_noise = rng.normal(size=(num_samples, self.coef.size)).astype(np.float32)
        weights = self.coef[None, :] + weight_noise * np.sqrt(self.var)[None, :]
        bias = self.intercept + rng.normal(size=(num_samples, 1)).astype(np.float32) * np.sqrt(self.bias_var)
        probs = expit(x @ weights.T + bias.T)
        p_mean = probs.mean(axis=1)
        predictive_entropy = binary_entropy(p_mean)
        expected_entropy = binary_entropy(probs).mean(axis=1)
        # Mutual information rises when sampled plausible probes disagree, a signal of epistemic uncertainty.
        mutual_information = predictive_entropy - expected_entropy
        return {
            "p_mean": p_mean.astype(np.float32),
            "predictive_entropy": predictive_entropy.astype(np.float32),
            "mutual_information": mutual_information.astype(np.float32),
        }
