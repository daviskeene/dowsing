import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from src.bayes_laplace import DiagonalLaplaceLogistic


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # TODO[DOWSING-04]: Fit feature scaling on training activations only.
    # Return float32 mean and std vectors over rows. Add 1e-6 to std so constant
    # activation dimensions do not divide by zero.
    raise NotImplementedError


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    # TODO[DOWSING-04]: Apply the train-set mean/std to any split.
    # Do not refit on val/test rows; that would leak evaluation information into
    # the probe. Return float32 because activation arrays can be large.
    raise NotImplementedError


def train_logistic(x_train: np.ndarray, y_train: np.ndarray, class_weight="balanced") -> LogisticRegression:
    # TODO[DOWSING-04]: Fit the deterministic probabilistic probe.
    # Use sklearn LogisticRegression with solver="lbfgs", max_iter high enough
    # for convergence, and the provided class_weight to handle imbalanced labels.
    raise NotImplementedError


def train_ridge(x_train: np.ndarray, y_train: np.ndarray) -> RidgeClassifier:
    # TODO[DOWSING-04]: Fit a simple linear margin baseline.
    # RidgeClassifier does not return calibrated probabilities, but its decision
    # function is useful as a cheap ranking score.
    raise NotImplementedError


def mahalanobis_fit(x_id: np.ndarray, ridge: float = 1e-3) -> dict[str, np.ndarray]:
    # TODO[DOWSING-04]: Fit the in-distribution activation cloud.
    # Use only negative/ID training rows, estimate their mean and covariance, add
    # ridge * I for numerical stability, and store the inverse covariance.
    raise NotImplementedError


def mahalanobis_score(x: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    # TODO[DOWSING-04]: Score rows by distance from the fitted ID cloud.
    # Larger Mahalanobis distance should mean "less like the training ID rows."
    raise NotImplementedError


def train_laplace_from_logistic(clf: LogisticRegression, x_train: np.ndarray, prior_precision: float = 1.0):
    # TODO[DOWSING-05]: Turn the fitted logistic probe into a Bayesian probe.
    # Initialize DiagonalLaplaceLogistic from clf.coef_ and clf.intercept_, then
    # fit its diagonal posterior on the same standardized training activations.
    raise NotImplementedError


def save_pickle(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path):
    with Path(path).open("rb") as f:
        return pickle.load(f)
