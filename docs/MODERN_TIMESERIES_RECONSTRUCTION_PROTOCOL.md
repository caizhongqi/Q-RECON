# Modern Time-Series Reconstruction Evidence Protocol

## 1. Purpose

This protocol turns Transformer, PatchTST and iTransformer support from a model
smoke test into a falsifiable reconstruction study. A model implementation or one
successful sample is not publication evidence. Every result must be generated
from a canonical manifest and must retain failed attacks in the denominator.

The current benchmark is a **classical white-box gradient-inversion experiment**.
It does not imply that the modern victim has been compiled into a coherent
quantum oracle.

## 2. Canonical manifest

`ModernTimeSeriesAttackManifest` fixes:

- dataset name, immutable revision or local path, selected columns, window length
  and horizon;
- victim architecture and every Transformer/PatchTST/iTransformer hyperparameter;
- training seed, optimizer, epochs, clipping and weight decay;
- attack batch start positions and batch size;
- all independent attack seeds;
- optimizer, steps, matching loss, layer weighting, regularization and clipping;
- exact tolerance and relative-L2 success threshold;
- confidence level, bootstrap seed and bootstrap sample count;
- publication completeness thresholds.

The canonical JSON representation has a SHA256 identity. The report separately
hashes the selected dataset tensors and the trained model state.

## 3. Best-iterate and restart rule

Gradient inversion is non-convex. Returning only the last iterate can turn a
transiently good reconstruction into a reported failure, while selecting by
reference MSE leaks the private answer into model selection.

Q-RECON therefore:

1. retains the minimum **released-gradient objective** reached during each run;
2. executes every declared independent restart;
3. selects the restart with minimum released-gradient objective;
4. uses the private reference only after selection to compute evaluation metrics.

Every failed restart remains in the completion-rate denominator with its error
type and a hash of the error message.

## 4. Batch permutation equivalence

Mean aggregate gradients are invariant to record order. For batch size greater
than one, ordered elementwise MSE is not a valid exact-recovery criterion.
Q-RECON solves the small-batch minimum-MSE assignment exactly by bitmask dynamic
programming and reports:

- the selected record assignment;
- aligned MSE and all standard reconstruction metrics;
- all-record exact success at the declared tolerance;
- per-record tolerance success.

The exact assignment implementation is capped at batch size 12. Larger batches
require a separately declared exact or approximate matching algorithm.

## 5. Required metrics

For every selected attack batch, report at least:

- MSE, MAE, RMSE and maximum absolute error;
- relative L2 error and correlation;
- tolerance percentages;
- best and final gradient-matching objective;
- best iteration and attack duration;
- exact batch success modulo permutation;
- record-level tolerance success.

Dataset-level summaries use Wilson intervals for success/completion proportions
and deterministic percentile-bootstrap intervals for scalar means and medians.

## 6. Publication completeness gate

The supplied publication templates require at least:

- 20 distinct attacked batches;
- 3 independent attack seeds per batch;
- a non-synthetic dataset;
- a modern victim (`transformer`, `patchtst` or `itransformer`);
- one successful selected attack for every declared batch;
- no silently discarded failed restart;
- `publication_mode=true`.

Passing this gate means the experiment matrix is complete under the manifest. It
does **not** mean the attack is effective, novel, or CCF-A ready.

## 7. Real-data templates

### GIFT-Eval with PatchTST

```bash
python examples/run_modern_timeseries_manifest.py \
  configs/modern_timeseries/gifteval_patchtst_publication.json \
  --output outputs/gifteval_patchtst_publication.json \
  --require-quality-gate
```

The template pins the GIFT-Eval dataset revision and attacks twenty independent
windows with three independent restarts each.

### ETTm1 with iTransformer

Place the official CSV at `data/ETT-small/ETTm1.csv`, then run:

```bash
python examples/run_modern_timeseries_manifest.py \
  configs/modern_timeseries/ettm1_itransformer_publication.json \
  --output outputs/ettm1_itransformer_publication.json \
  --require-quality-gate
```

The generic multivariate CSV adapter preserves variables, uses a declared ordered
column list, interpolates missing values deterministically, and normalizes each
window using context values only.

## 8. CI role

`examples/modern_timeseries_reconstruction_matrix.py` runs a deliberately tiny,
deterministic Transformer/PatchTST/iTransformer matrix in CI. It validates:

- training and gradient leakage;
- higher-order attack differentiation;
- independent restarts;
- objective-only selection;
- report serialization and confidence intervals.

Shared-runner smoke timing is not publication performance evidence. Final runtime
claims require pinned hardware, process isolation, warmups, repeated within-seed
measurements, affinity controls and raw artifacts.

## 9. Remaining CCF-A gate

A top-tier paper still needs all of the following on the same task:

1. a nonlinear identifiability/collision result for the modern leakage setting;
2. architecture-matched classical attacks, including optimized continuous and
   discrete solvers where applicable;
3. clipping, aggregation, partial-gradient, quantization and noise conditions;
4. a non-enumerative coherent access construction, or an explicit negative
   theorem showing why one cannot yield an advantage;
5. one common end-to-end cost unit with data loading, state preparation, inverse
   calls, diffusion, measurement, fault tolerance and uncertainty;
6. a robust nonempty advantage region, or a substantially generalized sharp
   no-advantage boundary.
