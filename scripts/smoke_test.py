import subprocess
import sys


COMMANDS = [
    [sys.executable, "data/uncertainty_byte/prepare.py", "--tiny"],
    [sys.executable, "train.py", "config/train_uncertainty_byte.py", "--max_iters=10", "--eval_interval=5", "--eval_iters=2", "--batch_size=8", "--device=cpu"],
    [sys.executable, "scripts/collect_activations.py", "--max_tokens_per_set=512", "--batch_size=4", "--device=cpu"],
    [sys.executable, "scripts/train_probes.py", "--task=ood", "--max_train_tokens=512", "--laplace_samples=16"],
    [sys.executable, "scripts/eval_probes.py", "--task=ood"],
    [sys.executable, "scripts/train_probes.py", "--task=high_loss", "--max_train_tokens=512", "--laplace_samples=16"],
    [sys.executable, "scripts/eval_probes.py", "--task=high_loss"],
    [sys.executable, "scripts/robustness_evals.py", "--laplace_samples=16"],
    [sys.executable, "scripts/make_plots.py"],
    [sys.executable, "scripts/analyze_results.py"],
    [sys.executable, "scripts/make_report_tables.py"],
]


def main() -> None:
    for command in COMMANDS:
        print("$ " + " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
