# Result tables

These CSVs are lightweight outputs from the current reference run.

Most useful files:

```text
high_loss_id_clean_only_summary.csv
ood_leave_one_type_out_summary.csv
fair_baselines.csv
ood_layerwise.csv
high_loss_layerwise.csv
```

The report reads these tables when generating:

```text
report/mini_report.md
```

Regenerate with:

```bash
make probes
make eval
make robustness
make plots
make report
```

