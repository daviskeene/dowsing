# Checkpoint 05: diagonal Laplace probe

## Goal

Implement the approximate Bayesian linear probe.

## Starter file

```text
lab/starter/src/bayes_laplace.py
```

## Main concepts

- MAP logistic regression weights
- diagonal posterior precision
- predictive mean
- predictive entropy
- expected entropy
- mutual information

## Check command

Fast function-level test:

```bash
.venv/bin/python lab/test.py --milestone 5
```

After copying your completed file:

```bash
.venv/bin/python scripts/train_probes.py --task ood --max_train_tokens 512 --laplace_samples 16
```

## Questions

- What does the diagonal approximation ignore?
- Why might residual-stream feature correlations matter?
- Why is the Bayesian result in DOWSING only a claim about this approximation?

## Completion criteria

You can train Laplace probes and explain why predictive mean and mutual information are different scores.
