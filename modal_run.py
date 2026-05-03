from __future__ import annotations

import argparse
import os
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "dowsing-nanogpt"
REMOTE_ROOT = Path("/root/dowsing")
VOLUME_ROOT = Path("/vol")
RESULT_VOLUME = modal.Volume.from_name("dowsing-results", create_if_missing=True)


def _ignore(path: Path) -> bool:
    # Ship source and configs to Modal, but rebuild datasets, checkpoints, and
    # analysis artifacts inside the remote run.
    parts = set(path.parts)
    name = path.name
    if name in {".venv", "__pycache__", ".DS_Store"}:
        return True
    if ".venv" in parts or "__pycache__" in parts:
        return True
    if "artifacts" in parts or "out-uncertainty-byte" in parts or "out-benchmark" in parts:
        return True
    if "modal_outputs" in parts:
        return True
    if name.endswith((".pyc", ".pyo")):
        return True
    if path.match("data/uncertainty_byte/*.bin"):
        return True
    if path.match("data/uncertainty_byte/eval/*.bin"):
        return True
    if "raw" in parts and "uncertainty_byte" in parts:
        return True
    return False


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .workdir(str(REMOTE_ROOT))
    .add_local_dir(".", remote_path=str(REMOTE_ROOT), ignore=_ignore)
)

app = modal.App(APP_NAME, image=image)


def _run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REMOTE_ROOT, env=env, check=True)


@app.function(
    gpu="L4",
    timeout=60 * 60 * 6,
    volumes={str(VOLUME_ROOT): RESULT_VOLUME},
)
def run_pipeline(
    run_name: str,
    mode: str = "full",
    max_iters: int = 3000,
    max_tokens_per_set: int = 50_000,
    eval_iters: int = 100,
    batch_size: int = 64,
    collect_batch_size: int = 32,
    laplace_samples: int = 64,
    include_activations: bool = False,
) -> str:
    import torch

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    _run(["python", "data/uncertainty_byte/prepare.py"], env=env)

    if mode in {"full", "train", "smoke"}:
        train_iters = 20 if mode == "smoke" else max_iters
        train_eval_interval = 10 if mode == "smoke" else 250
        train_eval_iters = 2 if mode == "smoke" else eval_iters
        _run(
            [
                "python",
                "train.py",
                "config/train_uncertainty_byte.py",
                f"--max_iters={train_iters}",
                f"--eval_interval={train_eval_interval}",
                f"--eval_iters={train_eval_iters}",
                f"--batch_size={batch_size}",
                "--device=cuda",
            ],
            env=env,
        )

    if mode in {"full", "collect", "smoke"}:
        collect_tokens = 1024 if mode == "smoke" else max_tokens_per_set
        _run(
            [
                "python",
                "scripts/collect_activations.py",
                "--ckpt",
                "out-uncertainty-byte/ckpt.pt",
                "--dataset",
                "data/uncertainty_byte",
                "--device",
                "cuda",
                "--max_tokens_per_set",
                str(collect_tokens),
                "--batch_size",
                str(collect_batch_size),
            ],
            env=env,
        )

    if mode in {"full", "probes", "smoke"}:
        probe_limit_args = ["--max_train_tokens", "2048"] if mode == "smoke" else []
        for task in ("ood", "high_loss"):
            _run(
                [
                    "python",
                    "scripts/train_probes.py",
                    "--task",
                    task,
                    "--laplace_samples",
                    str(16 if mode == "smoke" else laplace_samples),
                    *probe_limit_args,
                ],
                env=env,
            )
            _run(["python", "scripts/eval_probes.py", "--task", task], env=env)

    if mode in {"full", "plots", "probes", "smoke"}:
        _run(["python", "scripts/robustness_evals.py", "--laplace_samples", str(16 if mode == "smoke" else laplace_samples)], env=env)
        _run(["python", "scripts/make_plots.py"], env=env)
        _run(["python", "scripts/analyze_results.py"], env=env)
        _run(["python", "scripts/make_report_tables.py"], env=env)

    run_dir = VOLUME_ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tar_path = run_dir / "dowsing-results.tar.gz"
    # Keep the default bundle small; activation arrays are large and optional.
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in [
            "README.md",
            "LAB.md",
            "lab",
            "docs",
            "SPEC.md",
            "Makefile",
            "requirements.txt",
            "pyproject.toml",
            "config/train_uncertainty_byte.py",
            "out-uncertainty-byte/ckpt.pt",
            "artifacts/results",
            "artifacts/plots",
            "artifacts/probes",
            "report",
        ]:
            src = REMOTE_ROOT / path
            if src.exists():
                tar.add(src, arcname=path)
        if include_activations:
            src = REMOTE_ROOT / "artifacts/activations"
            if src.exists():
                tar.add(src, arcname="artifacts/activations")

    activation_marker = run_dir / "activation_artifacts_location.txt"
    activation_marker.write_text(
        "Activation arrays are included in dowsing-results.tar.gz only when "
        "`--include-activations` is set.\n",
        encoding="utf-8",
    )
    RESULT_VOLUME.commit()
    return str(tar_path.relative_to(VOLUME_ROOT))


@app.local_entrypoint()
def main(
    mode: str = "full",
    run_name: str = "",
    max_iters: int = 3000,
    max_tokens_per_set: int = 50_000,
    eval_iters: int = 100,
    batch_size: int = 64,
    collect_batch_size: int = 32,
    laplace_samples: int = 64,
    include_activations: bool = False,
    download_dir: str = "modal_outputs",
) -> None:
    if not run_name:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_name = f"{mode}-{stamp}"

    remote_tar = run_pipeline.remote(
        run_name=run_name,
        mode=mode,
        max_iters=max_iters,
        max_tokens_per_set=max_tokens_per_set,
        eval_iters=eval_iters,
        batch_size=batch_size,
        collect_batch_size=collect_batch_size,
        laplace_samples=laplace_samples,
        include_activations=include_activations,
    )

    local_dir = Path(download_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    local_tar = local_dir / f"{run_name}-dowsing-results.tar.gz"
    with local_tar.open("wb") as f:
        for chunk in RESULT_VOLUME.read_file(remote_tar):
            f.write(chunk)

    extract_dir = local_dir / run_name
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(extract_dir, filter="data")

    print(f"downloaded {local_tar}")
    print(f"extracted {extract_dir}")
