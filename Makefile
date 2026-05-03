.PHONY: setup prepare train collect probes eval robustness plots report smoke clean-cache all

PYTHON ?= python3

setup:
	$(PYTHON) -m pip install -r requirements.txt

prepare:
	$(PYTHON) data/uncertainty_byte/prepare.py

train:
	$(PYTHON) train.py config/train_uncertainty_byte.py

collect:
	$(PYTHON) scripts/collect_activations.py \
		--ckpt out-uncertainty-byte/ckpt.pt \
		--dataset data/uncertainty_byte

probes:
	$(PYTHON) scripts/train_probes.py --task ood
	$(PYTHON) scripts/train_probes.py --task high_loss

eval:
	$(PYTHON) scripts/eval_probes.py --task ood
	$(PYTHON) scripts/eval_probes.py --task high_loss

robustness:
	$(PYTHON) scripts/robustness_evals.py

plots:
	$(PYTHON) scripts/make_plots.py
	$(PYTHON) scripts/analyze_results.py

report:
	$(PYTHON) scripts/make_report_tables.py

smoke:
	$(PYTHON) scripts/smoke_test.py

clean-cache:
	find . \( -path './.venv' -o -path './.venv/*' \) -prune -o -type d -name '__pycache__' -print -exec rm -rf {} +
	find . \( -path './.venv' -o -path './.venv/*' \) -prune -o -type f -name '*.py[co]' -print -delete

all: prepare train collect probes eval robustness plots report
