# DOWSING: Detecting OOD and Weird States in NanoGPT

**White-box uncertainty monitoring in tiny transformers**

## Abstract

We train a small byte-level nanoGPT model on Tiny Shakespeare and test whether frozen internal activations contain signals predictive of high-loss and distribution-shifted tokens. We compare output-level confidence metrics against deterministic and diagonal-Laplace Bayesian probes trained on layerwise hidden states. The cleanest positive result is within-distribution: a held-out Shakespeare-only high-loss probe at layer 4 reaches 0.779 AUROC vs entropy 0.656. The known-shift aggregate OOD probe reaches 0.879 AUROC vs entropy 0.566 but generalizes to only 4 of 5 held-out shift families under leave-one-shift-out, failing notably on char_shuffle. On the held-out shifts the logistic probe handles worst, an OOD-free Mahalanobis detector on early-layer activations generalizes better. Diagonal-Laplace Bayesian uncertainty does not improve over the deterministic logistic readout in this implementation, conditional on the diagonal posterior approximation.

## Hypotheses

H1: Layerwise activation probes predict high-loss tokens better than output entropy.
H2: Layerwise activation probes detect some OOD shifts better than output entropy.
H3: Probe performance varies by layer.
H4: Bayesian uncertainty provides additional signal beyond deterministic probe scores.

## Method

The harness trains a 6-layer byte-level nanoGPT model on Tiny Shakespeare, reserves held-out Shakespeare for ID evaluation, and builds synthetic Python, modern prose, shuffled Shakespeare, and repetition OOD sets. During evaluation it records per-token loss, entropy, max probability, target probability, model top prediction, and every block activation. Probes are trained on frozen post-block residual activations with sequence-level splits, so adjacent tokens from the same fixed window do not leak across train/test splits.

High-loss labels are defined by the top 20% of held-out ID losses. Target probability is therefore an oracle diagnostic for high-loss, because token loss is `-log(target probability)`. It is excluded from fair high-loss baseline comparisons.

The aggregate OOD probe is a known-shift detector: training includes examples from the same OOD families used at test time. To separate memorization of known shifts from generalization to unseen shifts, the robustness section adds a leave-one-OOD-type-out evaluation. The Python, modern prose, and repetition sets are template-generated and repeated to fill the eval split, so those rows should be read as engineering stress tests rather than natural distribution estimates. The controlled `char_shuffle` and `word_shuffle` sets are more load-bearing because they are derived from held-out Shakespeare bytes rather than tiled snippets.

## Run Metadata

- Compute: Modal NVIDIA L4 GPU
- Training: 3,000 iterations, batch size 64, block size 128
- Model: 6 layers, 6 heads, 192 embedding dimensions, byte vocabulary size 256
- Evaluation: 50,000 target tokens per eval set
- Final logged validation loss: approximately 1.444
- Artifact bundle: `modal_outputs/full-l4-3000-robustness-dowsing-results.tar.gz`

## Results

### Known-Shift Result

When the probe is trained on examples from all five OOD families, the best layer probe reached 0.879 AUROC, compared with entropy at 0.566 AUROC and loss at 0.681 AUROC. The improvement over entropy is +0.313 AUROC. This is a useful known-shift monitoring result, not a claim of general OOD detection.

For the original high-loss task evaluated over ID plus OOD rows, the best layer probe reached 0.797 AUROC, compared with entropy at 0.627 AUROC and negative max probability at 0.608 AUROC. This number is partly redundant with OOD detection because OOD rows are systematically higher loss; the ID-only high-loss robustness check below is the cleaner token-surprise test.

### Hypothesis Verdicts

| hypothesis | verdict | evidence |
| --- | --- | --- |
| H1: probes predict high-loss better than entropy | Supported | ID-only high-loss: probe layer 4 0.779 AUROC vs entropy 0.656 |
| H2: probes detect OOD better than entropy | Mixed under LOSO: probe beats entropy on 4/5 held-out shifts | known-shift aggregate: probe 0.879 vs entropy 0.566; LOSO char_shuffle delta -0.013 (probe underperforms entropy); non-logistic best on char_shuffle:mahalanobis, modern_prose:ridge_classifier, python:mahalanobis, word_shuffle:ridge_classifier |
| H3: probe performance varies by layer | Supported | best OOD layer 5; best high-loss layer 5 |
| H4: Bayesian uncertainty adds signal | Not supported (diagonal Laplace approximation) | Laplace mean matches logistic; Laplace MI performs poorly; result conditional on diagonal posterior |

### Best OOD Layerwise Results

| task | probe_type | layer | auroc | auprc | brier | ece |
| --- | --- | --- | --- | --- | --- | --- |
| ood | deterministic_logistic | 5 | 0.8790 | 0.9708 | 0.1373 | 0.1760 |
| ood | laplace_predictive_mean | 5 | 0.8790 | 0.9708 | 0.1373 | 0.1760 |
| ood | ridge_classifier | 5 | 0.8780 | 0.9705 |  |  |
| ood | mahalanobis | 1 | 0.6290 | 0.8937 |  |  |
| ood | activation_norm | 0 | 0.4861 | 0.8430 |  |  |
| ood | laplace_mutual_information | 1 | 0.4552 | 0.8185 |  |  |

### Best High-Loss Layerwise Results

| task | probe_type | layer | auroc | auprc | brier | ece |
| --- | --- | --- | --- | --- | --- | --- |
| high_loss | ridge_classifier | 5 | 0.7974 | 0.7204 |  |  |
| high_loss | deterministic_logistic | 5 | 0.7973 | 0.7265 | 0.1842 | 0.0624 |
| high_loss | laplace_predictive_mean | 5 | 0.7972 | 0.7263 | 0.1843 | 0.0625 |
| high_loss | mahalanobis | 0 | 0.6997 | 0.6225 |  |  |
| high_loss | laplace_mutual_information | 1 | 0.5058 | 0.4271 |  |  |
| high_loss | activation_norm | 0 | 0.4931 | 0.4814 |  |  |

### Fair Baseline Comparison

| task | method | layer | auroc | auprc | brier | ece | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ood | entropy |  | 0.5657 | 0.8583 |  |  | output baseline |
| ood | neg_max_prob |  | 0.5558 | 0.8483 |  |  | output baseline |
| ood | neg_target_prob |  | 0.6812 | 0.9173 |  |  | output baseline |
| ood | loss |  | 0.6812 | 0.9173 |  |  | output baseline |
| ood | activation_norm | 0 | 0.4861 | 0.8430 |  |  | best layer |
| ood | mahalanobis | 1 | 0.6290 | 0.8937 |  |  | best layer |
| ood | deterministic_logistic | 5 | 0.8790 | 0.9708 | 0.1373 | 0.1760 | best layer |
| ood | laplace_predictive_mean | 5 | 0.8790 | 0.9708 | 0.1373 | 0.1760 | best layer |
| ood | laplace_mutual_information | 1 | 0.4552 | 0.8185 |  |  | best layer |
| high_loss | entropy |  | 0.6266 | 0.5424 |  |  | output baseline |
| high_loss | neg_max_prob |  | 0.6084 | 0.5195 |  |  | output baseline |
| high_loss | activation_norm | 0 | 0.4931 | 0.4814 |  |  | best layer |
| high_loss | mahalanobis | 0 | 0.6997 | 0.6225 |  |  | best layer |
| high_loss | deterministic_logistic | 5 | 0.7973 | 0.7265 | 0.1842 | 0.0624 | best layer |
| high_loss | laplace_predictive_mean | 5 | 0.7972 | 0.7263 | 0.1843 | 0.0625 | best layer |
| high_loss | laplace_mutual_information | 1 | 0.5058 | 0.4271 |  |  | best layer |

Calibration metrics are reported only for probability outputs. Raw distances, norms, margins, entropy, and loss are ranked scores, so their Brier, NLL, and ECE entries are intentionally blank.

Because logistic regression is trained with class weighting, its probabilities should be interpreted as scoring outputs rather than deployment-calibrated forecasts. AUROC and AUPRC are the primary metrics; calibration metrics are secondary diagnostics only.

### Robustness: Leave-One-OOD-Type-Out

| holdout_ood_type | best_probe_layer | probe_auroc | entropy_auroc | loss_auroc | delta_vs_entropy |
| --- | --- | --- | --- | --- | --- |
| modern_prose | 2 | 0.8217 | 0.5711 | 0.5851 | 0.2506 |
| repetition | 5 | 0.6901 | 0.5049 | 0.4972 | 0.1852 |
| word_shuffle | 4 | 0.7288 | 0.5536 | 0.6290 | 0.1751 |
| python | 5 | 0.7469 | 0.5908 | 0.7827 | 0.1561 |
| char_shuffle | 5 | 0.5898 | 0.6032 | 0.8983 | -0.0134 |

This evaluation trains the probe on ID plus four OOD families and tests it on the held-out fifth family plus held-out ID. It is a stricter test than the aggregate known-shift result because the target shift type is not present in probe training. The trained logistic probe beats entropy on 4 of 5 held-out shifts. The exception is `char_shuffle`, where the probe underperforms entropy (-0.013 AUROC). This is the most damaging row in the report: char_shuffle was previously presented as the most controlled sanity check, and it is the one shift that the trained probe does not generalize to when held out of training. The known-shift char_shuffle gain therefore reflects in-distribution memorization of that shift family rather than a general OOD signal.

### Robustness: Best Detector Per Held-Out Shift

| holdout_ood_type | best_method | best_layer | best_auroc |
| --- | --- | --- | --- |
| char_shuffle | mahalanobis | 0 | 0.6957 |
| modern_prose | ridge_classifier | 2 | 0.8288 |
| python | mahalanobis | 1 | 0.8581 |
| repetition | deterministic_logistic | 5 | 0.6901 |
| word_shuffle | ridge_classifier | 4 | 0.7389 |

This table ranks only deployable detectors. The `loss` and `neg_target_prob` baselines are excluded because they require knowing the true next token; they appear in the LOSO summary above for reference but are oracle diagnostics, not monitors.

When the held-out shift is genuinely unseen, the trained logistic probe is not always the best deployable detector. Mahalanobis distance from the ID training mean — a one-class density estimator that is *not* trained on any OOD examples — beats the logistic probe on the two LOSO settings where the probe struggles most. On char_shuffle, Mahalanobis at layer 0 reaches 0.696 AUROC vs probe 0.590; on python, Mahalanobis at layer 1 reaches 0.858 vs probe 0.747. This suggests that for unseen shift types, an OOD-free density model on early-layer activations generalizes better than a trained linear readout, and that the right detector depends on the shift.

### Robustness: ID-Only High-Loss

| method | layer | auroc | auprc | brier | ece | n_test |
| --- | --- | --- | --- | --- | --- | --- |
| deterministic_logistic | 4 | 0.7789 | 0.5170 | 0.1842 | 0.2207 | 9960 |
| laplace_predictive_mean | 4 | 0.7786 | 0.5168 | 0.1841 | 0.2212 | 9960 |
| entropy |  | 0.6563 | 0.3207 |  |  | 9960 |
| neg_max_prob |  | 0.6384 | 0.2980 |  |  | 9960 |
| mahalanobis | 0 | 0.5939 | 0.3463 |  |  | 9960 |
| standardized_activation_norm | 0 | 0.5058 | 0.2335 |  |  | 9960 |

This evaluation trains and tests only on held-out Shakespeare windows. The high-loss threshold is fit from ID train rows only, so the task measures within-distribution token surprise rather than dataset shift.

### OOD Type Breakdown

| ood_type | layer5_probe_auroc | entropy_auroc | delta |
| --- | --- | --- | --- |
| repetition | 0.9621 | 0.5049 | 0.4571 |
| modern_prose | 0.9005 | 0.5711 | 0.3293 |
| python | 0.9198 | 0.5908 | 0.3290 |
| word_shuffle | 0.8322 | 0.5536 | 0.2785 |
| char_shuffle | 0.7744 | 0.6032 | 0.1712 |

The OOD-type breakdown tests whether the aggregate known-shift result is driven only by easy shifts such as Python code or repetition. Within the known-shift task, the layer-5 probe beats entropy on every shift, including controlled shuffled-Shakespeare. However, this section is **conditional on having seen each shift family during probe training**: the leave-one-shift-out section above shows that the char_shuffle gain in particular does not survive when char_shuffle is withheld from training. Read these per-type numbers as known-shift monitoring strength, not as evidence of general OOD detection.

The table reports the layer-5 probe for each OOD type, matching the best aggregate OOD layer. A separate layer-by-type sweep shows that Python and modern prose are most detectable in earlier or middle layers, while repetition, word shuffle, and char shuffle are strongest in later layers. This suggests that different distribution shifts become linearly accessible at different depths.

### Best Layer By OOD Type

| ood_type | best_layer | best_probe_auroc |
| --- | --- | --- |
| char_shuffle | 5 | 0.7744 |
| modern_prose | 2 | 0.9025 |
| python | 1 | 0.9433 |
| repetition | 5 | 0.9621 |
| word_shuffle | 5 | 0.8322 |

On `char_shuffle`, next-token loss reaches 0.898 AUROC, above the layer-5 probe at 0.774. This suggests that the corruption is directly reflected in target-token surprise. However, loss requires knowledge of the correct next token and is therefore an oracle-style retrospective diagnostic rather than a generation-time monitor.

## Key Figures

- `artifacts/plots/ood_auroc_by_layer.png`
- `artifacts/plots/high_loss_auroc_by_layer.png`
- `artifacts/plots/ood_type_breakdown.png`
- `artifacts/plots/ood_probe_delta_by_type.png`
- `artifacts/plots/ood_layer_type_heatmap.png`
- `artifacts/plots/confident_failures_probe_score.png`

## Qualitative Examples

| case | set_name | target | top_prediction | loss | entropy | probe_score |
| --- | --- | --- | --- | --- | --- | --- |
| probe_catch | ood_char_shuffle | 't' | 'E' | 18.3832 | 0.0157 | 0.9579 |
| probe_catch | ood_python | ')' | '\n' | 20.8188 | 0.7531 | 0.8707 |
| probe_catch | ood_python | ')' | ' ' | 19.8697 | 0.6509 | 0.8602 |
| probe_catch | ood_python | ')' | ' ' | 17.4366 | 0.8204 | 0.8568 |
| probe_catch | ood_python | ')' | ' ' | 17.0368 | 0.9253 | 0.8533 |
| probe_miss | ood_python | '=' | ' ' | 24.1739 | 1.0246 | 0.3843 |
| probe_miss | ood_python | '=' | ' ' | 23.7180 | 1.0629 | 0.3475 |
| probe_miss | ood_python | '(' | ' ' | 22.3870 | 0.4255 | 0.1725 |
| probe_miss | ood_python | '(' | ' ' | 22.2653 | 0.5151 | 0.1801 |
| probe_miss | ood_python | '(' | ' ' | 22.1656 | 0.6315 | 0.1839 |

The char-shuffle example shows the kind of confident-failure token output entropy misses. Important caveat: the probe used here was trained on examples from `ood_char_shuffle`, so this is an in-distribution catch under the known-shift task, not a generalization result. The leave-one-shift-out evaluation above shows the trained probe does not generalize to char_shuffle when that family is held out of training; under that stricter setup, layer-0 Mahalanobis is the better detector for this kind of corruption. This example illustrates the failure-mode geometry, not white-box monitoring of unseen shifts.

## Interpretation

The defensible claims are narrower than the known-shift headline suggests.

**Within-distribution token surprise (H1).** The cleanest result is the ID-only high-loss check. Trained on held-out Shakespeare windows alone, with the high-loss threshold fit from ID train rows only, the layer-4 logistic probe reaches 0.779 AUROC vs entropy 0.656. This is a real signal about token-level surprise that is not confounded with dataset shift, and it survives the audit corrections.

**Known-shift OOD (H2, in-distribution).** Trained on examples from all five OOD families, the layer-5 probe reaches 0.879 AUROC vs entropy 0.566. This is a useful known-shift monitoring result. It is not evidence of a general OOD signal: the probe has seen examples of every shift family during training.

**Leave-one-shift-out OOD (H2, generalization).** The trained probe beats entropy on 4 of 5 held-out shifts. It fails on char_shuffle (-0.013 AUROC vs entropy), which had been presented as the most controlled sanity check. On char_shuffle and python — the two hardest LOSO settings for the logistic probe — Mahalanobis distance from the ID-training mean (a one-class detector that uses no OOD examples) generalizes better than the trained probe. The honest summary is that activations carry shift-relevant information, but which detector exposes it depends on whether the shift family was seen in training. The known-shift gain on char_shuffle reflects memorization of that family, not a general signal.

**Bayesian uncertainty (H4).** Negative for the specific approximation tested: diagonal-Laplace predictive mean is effectively identical to the deterministic logistic readout, and Laplace MI performs near chance. Residual-stream features are correlated, and a diagonal posterior with fixed prior precision is a severe approximation, so this is a result about *this approximation*, not about Bayesian uncertainty more generally. A full or low-rank posterior could change the picture.

**Caveats.** The probes are correlational; the model is tiny, undertrained (validation loss ~1.444), and run with a single LM seed and a single split seed; three of the OOD sets are tiled templates that admit content memorization across train/test windows. There are no error bars. The strongest claim the data support is: in this tiny-transformer setting, frozen activations carry within-distribution token-surprise signal beyond entropy, and carry known-shift detection signal that mostly but not fully generalizes to unseen shift families.

## Limitations

- Tiny model
- Byte-level tokenization
- Synthetic OOD distributions
- Python, prose, and repetition OOD sets are tiled templates, so they permit content memorization
- Aggregate OOD detection is a known-shift task unless using the leave-one-type-out robustness table
- Original high-loss detection is partly confounded with OOD; ID-only high-loss is cleaner
- Approximate diagonal Bayesian probe with fixed prior precision
- Single LM seed and single split seed; no error bars or bootstrap confidence intervals
- Post-block residual hooks only; no embedding or final layer-norm probe
- Position-in-window effects are not controlled
- Probes are correlational, not causal

## Summary Claim

DOWSING shows that, in this tiny-transformer setting, frozen nanoGPT activations carry two kinds of signal beyond output entropy: a within-distribution token-surprise signal (ID-only high-loss probe layer 4, 0.779 AUROC vs entropy 0.656), and a known-shift detection signal that generalizes to 4 of 5 held-out shift families but not to controlled char-shuffle when that family is withheld from training. On the held-out shifts where the trained probe struggles, an OOD-free Mahalanobis density estimator on early-layer activations is the better detector, suggesting the right white-box monitor is shift-dependent. Diagonal-Laplace Bayesian uncertainty did not add signal beyond the deterministic readout in this implementation, but this is contingent on the diagonal posterior approximation rather than a general statement about Bayesian uncertainty.

## Future Work

- Activation interventions along probe directions
- Replace tiled synthetic OOD with sampled corpora and de-duplicate windows across splits
- Report leave-one-shift-out as the primary OOD metric
- Evaluate high-loss within ID and within each OOD family separately
- Add bootstrap confidence intervals and repeat over language-model and split seeds
- Larger datasets and GPT-2 BPE variants
- Comparison against sparse autoencoder features
- Generation-time monitoring
