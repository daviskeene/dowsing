#!/usr/bin/env python3
"""Test bench for the DOWSING fill-in lab.

These are small milestone checks, not a full experiment run. They exercise the
starter TODOs with tiny synthetic inputs so learners can get fast feedback while
building the data, activation, metric, probe, and robustness pieces.

Run examples:

    python lab/test.py --milestone 1
    python lab/test.py --milestone 5
    python lab/test.py --all
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "lab" / "starter"
sys.path.insert(0, str(STARTER))
sys.path.insert(1, str(ROOT))


class LabFailure(AssertionError):
    pass


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_close(actual, expected, tol=1e-6, msg="values differ") -> None:
    if not np.allclose(actual, expected, atol=tol, rtol=tol, equal_nan=True):
        raise LabFailure(f"{msg}: got {actual!r}, expected {expected!r}")


def assert_true(condition, msg: str) -> None:
    if not condition:
        raise LabFailure(msg)


def milestone_1_data() -> None:
    prepare = load_module("lab_prepare", STARTER / "data" / "uncertainty_byte" / "prepare.py")

    text = "Hello, π!\n"
    ids = prepare.encode(text)
    assert_true(ids.dtype == np.uint16, "encode should return uint16 IDs")
    assert_true(ids.min() >= 0 and ids.max() <= 255, "byte IDs should be in [0, 255]")
    assert_true(prepare.decode(ids) == text, "decode(encode(text)) should round-trip")

    train, val, heldout = prepare.split_text("abcdefghij", tiny=False)
    assert_true((train, val, heldout) == ("abcdefgh", "i", "j"), "split_text should use 80/10/10 splits")

    shuffled = prepare.char_shuffle("abcdefghijklmnopqrstuvwxyz", window=5, seed=1)
    assert_true(len(shuffled) == 26, "char_shuffle should preserve length")
    assert_true(sorted(shuffled[:5]) == sorted("abcde"), "char_shuffle should only shuffle inside each window")

    word_shuffled = prepare.word_shuffle("one two three four five six", window=3, seed=1)
    assert_true(sorted(word_shuffled.replace("\n", " ").split()) == ["five", "four", "one", "six", "three", "two"], "word_shuffle should preserve words")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.bin"
        prepare.save_bin(path, "abc")
        loaded = np.fromfile(path, dtype=np.uint16)
        assert_true(loaded.tolist() == [97, 98, 99], "save_bin should write byte IDs as uint16")


class TinyBlock(nn.Module):
    def forward(self, x):
        return x + 1


class TinyHookModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = type("Transformer", (), {})()
        self.transformer.h = nn.ModuleList([TinyBlock(), TinyBlock()])

    def forward(self, x):
        for block in self.transformer.h:
            x = block(x)
        return x


def milestone_2_activation_capture() -> None:
    capture = load_module("lab_activation_capture", STARTER / "src" / "activation_capture.py")

    model = TinyHookModel()
    activations, handles = capture.register_block_hooks(model)
    x = torch.zeros(2, 3, 4)
    out = model(x)
    capture.remove_hooks(handles)
    assert_true(torch.equal(out, torch.full_like(x, 2.0)), "tiny hook model should add two")
    assert_true(set(activations.keys()) == {0, 1}, "hooks should capture both block outputs")
    assert_true(tuple(activations[0].shape) == (2, 3, 4), "captured activation shape should be [B,T,C]")
    assert_true(not activations[0].requires_grad, "captured activations should be detached")

    logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 3.0, 1.0]]])
    targets = torch.tensor([[0, 2]])
    metrics = capture.per_token_metrics(logits, targets)
    expected_keys = {"loss", "entropy", "max_prob", "target_prob", "top_pred_id", "top_pred_prob"}
    assert_true(expected_keys.issubset(metrics.keys()), f"per_token_metrics missing keys: {expected_keys - set(metrics)}")
    for key in expected_keys:
        assert_true(tuple(metrics[key].shape) == (1, 2), f"{key} should have shape [B,T]")
    expected_loss = torch.nn.functional.cross_entropy(logits.view(-1, 3), targets.view(-1), reduction="none").view(1, 2)
    assert_close(metrics["loss"].detach().numpy(), expected_loss.detach().numpy(), msg="per-token loss mismatch")
    assert_true(metrics["top_pred_id"].tolist() == [[0, 1]], "top_pred_id should be argmax over vocab")


def milestone_3_metrics() -> None:
    metrics_mod = load_module("lab_metrics", STARTER / "src" / "metrics.py")
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.4, 0.35, 0.9])
    prob = np.array([0.05, 0.25, 0.8, 0.9])
    out = metrics_mod.binary_metrics(y, score, prob=prob)
    assert_close(out["auroc"], roc_auc_score(y, score), msg="AUROC mismatch")
    assert_close(out["auprc"], average_precision_score(y, score), msg="AUPRC mismatch")
    assert_true(np.isfinite(out["brier"]), "Brier should be finite when prob is provided")
    assert_true(np.isfinite(out["nll"]), "NLL should be finite when prob is provided")
    assert_true(0.0 <= out["ece"] <= 1.0, "ECE should be in [0,1]")

    raw = metrics_mod.binary_metrics(y, score, prob=None)
    assert_true(np.isnan(raw["brier"]) and np.isnan(raw["nll"]) and np.isnan(raw["ece"]), "calibration metrics should be NaN for raw scores")
    assert_true(np.isnan(metrics_mod.safe_auc(np.zeros(4), score)), "single-class AUROC should be NaN")


def milestone_4_probes() -> None:
    probes = load_module("lab_probes", STARTER / "src" / "probes.py")
    rng = np.random.default_rng(0)
    x_neg = rng.normal(loc=-1.0, scale=0.3, size=(30, 4))
    x_pos = rng.normal(loc=1.0, scale=0.3, size=(30, 4))
    x = np.vstack([x_neg, x_pos]).astype(np.float32)
    y = np.array([0] * 30 + [1] * 30)

    mean, std = probes.standardize_fit(x[:40])
    xz = probes.standardize_apply(x, mean, std)
    assert_true(mean.shape == (4,) and std.shape == (4,), "standardize_fit should return feature vectors")
    assert_true(xz.dtype == np.float32, "standardized activations should be float32")

    clf = probes.train_logistic(xz[:40], y[:40])
    probs = clf.predict_proba(xz[40:])[:, 1]
    assert_true(probs.shape == (20,), "logistic predict_proba should score test rows")

    ridge = probes.train_ridge(xz[:40], y[:40])
    margins = ridge.decision_function(xz[40:])
    assert_true(margins.shape == (20,), "ridge decision_function should score test rows")

    maha = probes.mahalanobis_fit(xz[y == 0])
    distances = probes.mahalanobis_score(xz, maha)
    assert_true(distances.shape == (60,), "Mahalanobis score should return one distance per row")
    assert_true(distances[y == 1].mean() > distances[y == 0].mean(), "positive cluster should be farther from negative ID mean")


def milestone_5_bayesian_probe() -> None:
    probes = load_module("lab_probes_for_laplace", STARTER / "src" / "probes.py")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 5)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    mean, std = probes.standardize_fit(x)
    xz = probes.standardize_apply(x, mean, std)
    clf = probes.train_logistic(xz, y)
    laplace = probes.train_laplace_from_logistic(clf, xz)
    out = laplace.predict(xz[:10], num_samples=16, seed=1)
    expected = {"p_mean", "predictive_entropy", "expected_entropy", "mutual_information"}
    assert_true(expected.issubset(out.keys()), f"Laplace predict missing keys: {expected - set(out)}")
    for key in expected:
        assert_true(out[key].shape == (10,), f"{key} should have one value per row")
    assert_true(np.all((out["p_mean"] >= 0) & (out["p_mean"] <= 1)), "p_mean should be probabilities")
    assert_true(np.all(out["mutual_information"] >= -1e-5), "MI should be nonnegative up to numerical noise")


def milestone_6_robustness() -> None:
    robust = load_module("lab_robustness", STARTER / "scripts" / "robustness_evals.py")
    metrics = pd.DataFrame(
        {
            "set_name": ["id_clean", "id_clean", "ood_python", "ood_python", "ood_char_shuffle", "ood_char_shuffle"],
            "sequence_id": [0, 1, 0, 1, 0, 1],
            "ood_type": ["id_clean", "id_clean", "python", "python", "char_shuffle", "char_shuffle"],
            "is_ood": [0, 0, 1, 1, 1, 1],
            "loss": [0.1, 2.0, 3.0, 4.0, 2.5, 5.0],
            "entropy": [0.2, 0.8, 0.9, 0.7, 0.3, 0.6],
            "max_prob": [0.9, 0.5, 0.4, 0.6, 0.8, 0.7],
        }
    )
    splits = robust.split_masks(metrics)
    assert_true(set(splits.keys()) == {"train", "val", "test"}, "split_masks should return train/val/test")
    for name, mask in splits.items():
        assert_true(mask.dtype == bool and mask.shape == (len(metrics),), f"{name} mask should be boolean and row-aligned")
    combined = splits["train"].astype(int) + splits["val"].astype(int) + splits["test"].astype(int)
    assert_true(np.all(combined == 1), "each row should appear in exactly one split")


MILESTONES = {
    1: ("byte-level data prep", milestone_1_data),
    2: ("activation capture", milestone_2_activation_capture),
    3: ("metrics", milestone_3_metrics),
    4: ("deterministic probes", milestone_4_probes),
    5: ("diagonal Laplace probe", milestone_5_bayesian_probe),
    6: ("robustness split helpers", milestone_6_robustness),
}


def run_milestone(number: int) -> bool:
    title, fn = MILESTONES[number]
    print(f"milestone {number}: {title}")
    try:
        fn()
    except NotImplementedError:
        print("  FAIL: hit NotImplementedError. Fill the TODOs for this milestone.")
        return False
    except Exception as exc:
        print(f"  FAIL: {exc}")
        traceback.print_exc(limit=2)
        return False
    print("  PASS")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone", type=int, choices=sorted(MILESTONES))
    parser.add_argument("--all", action="store_true", help="run all milestone tests")
    args = parser.parse_args()

    if not args.all and args.milestone is None:
        parser.error("choose --milestone N or --all")

    if args.all:
        ok = True
        for number in sorted(MILESTONES):
            ok = run_milestone(number) and ok
        raise SystemExit(0 if ok else 1)

    raise SystemExit(0 if run_milestone(args.milestone) else 1)


if __name__ == "__main__":
    main()
