from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from src.bayes_laplace import DiagonalLaplaceLogistic


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    std = (x.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) - mean) / std).astype(np.float32)


def train_logistic(x_train: np.ndarray, y_train: np.ndarray, class_weight="balanced") -> LogisticRegression:
    clf = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight=class_weight,
    )
    clf.fit(x_train, y_train)
    return clf


def train_ridge(x_train: np.ndarray, y_train: np.ndarray) -> RidgeClassifier:
    clf = RidgeClassifier(class_weight="balanced")
    clf.fit(x_train, y_train)
    return clf


def mahalanobis_fit(x_id: np.ndarray, ridge: float = 1e-3) -> dict[str, np.ndarray]:
    mean = x_id.mean(axis=0)
    cov = np.cov(x_id, rowvar=False)
    cov = cov + ridge * np.eye(cov.shape[0], dtype=cov.dtype)
    inv_cov = np.linalg.pinv(cov)
    return {"mean": mean.astype(np.float32), "inv_cov": inv_cov.astype(np.float32)}


def mahalanobis_score(x: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    return cdist(x, params["mean"][None, :], metric="mahalanobis", VI=params["inv_cov"]).reshape(-1)


def train_laplace_from_logistic(clf: LogisticRegression, x_train: np.ndarray, prior_precision: float = 1.0):
    laplace = DiagonalLaplaceLogistic(clf.coef_[0], clf.intercept_[0], prior_precision=prior_precision)
    return laplace.fit_posterior(x_train)


def save_pickle(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path):
    with Path(path).open("rb") as f:
        return pickle.load(f)
