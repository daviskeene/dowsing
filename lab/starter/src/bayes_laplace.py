import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def binary_entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


class DiagonalLaplaceLogistic:
    def __init__(self, weight_mean, bias_mean, prior_precision: float = 1.0):
        self.weight_mean = np.asarray(weight_mean, dtype=np.float32)
        self.bias_mean = float(bias_mean)
        self.prior_precision = float(prior_precision)
        self.weight_var = None
        self.bias_var = None

    def fit_posterior(self, x_train: np.ndarray):
        """Fit diagonal Laplace posterior around MAP logistic weights."""
        # TODO[DOWSING-05]: Estimate uncertainty around the fitted logistic weights.
        # Compute logits from the current weight_mean/bias_mean, convert to
        # probabilities p, then use p * (1 - p) as the local curvature for each
        # training row. For feature j:
        #   precision_j = prior_precision + sum_i p_i(1-p_i) * x_ij^2
        # Store weight_var = 1 / precision and do the analogous scalar update for
        # bias_var. Return self so callers can chain fit_posterior().
        raise NotImplementedError

    def predict(self, x: np.ndarray, num_samples: int = 64, seed: int = 1337) -> dict[str, np.ndarray]:
        """Sample weights, average probabilities, and return Bayesian uncertainty scores."""
        # TODO[DOWSING-05]: Sample plausible probes from the diagonal posterior.
        # Draw num_samples weights and biases using weight_var/bias_var, compute
        # probabilities for each sample, then average them into p_mean.
        # predictive_entropy measures uncertainty after averaging; expected_entropy
        # averages each sample's own uncertainty. Their difference is mutual
        # information, which is high when plausible probes disagree.
        raise NotImplementedError
