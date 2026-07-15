# Q-RECON Modern-Model Experimental Evidence Update — 2026-07-15

## Scope

This record freezes the modern time-series evidence produced after the initial
CCF-A evidence ledger. It is an execution and claim audit, not a declaration that
CCF-A acceptance or practical quantum advantage has been achieved.

The primary supported claim is negative and information-theoretic:
anonymous-channel equivariant forecasting models induce exact channel-permutation
training-data fibres. Classical and coherent quantum procedures receive the same
observation on every orbit member unless additional channel-identity side
information is supplied.

## 1. Large-model, long-context stress replication

- Workflow run: `29411342735`
- Artifact: `ett-large-model-channel-fibre-stress`
- Artifact id: `8341371469`
- Artifact digest:
  `sha256:aa9b4d2ca170d076d7a301f9de5221f6637e484089f425b2b50016328e755925`
- Quality gate: passed

The study covers three immutable real datasets and two modern architecture
families:

1. ETTm1 × iTransformer, `d_model=32`, two encoder layers;
2. ETTm1 × shared-head PatchTST, `d_model=32`, three encoder layers;
3. ETTm2 × iTransformer;
4. ETTm2 × shared-head PatchTST;
5. ETTh1 × iTransformer;
6. ETTh1 × shared-head PatchTST.

Each cell contains ten windows with context 96 and horizon 24, giving 60 primary
observations. All generator-complete fibre gates pass. The mean orbit size is
`5040` in every cell, yielding the uniform exact labeled-order ceiling
`1/5040`.

Maximum observed float32 numerical discrepancies are:

| dataset | architecture | max gradient discrepancy | max output discrepancy |
|---|---|---:|---:|
| ETTm1 | iTransformer | `2.0862e-7` | `5.3644e-7` |
| ETTm1 | PatchTST | `2.3842e-7` | `7.7486e-7` |
| ETTm2 | iTransformer | `2.3842e-7` | `5.9605e-7` |
| ETTm2 | PatchTST | `1.1921e-7` | `8.3447e-7` |
| ETTh1 | iTransformer | `1.7881e-7` | `6.2585e-7` |
| ETTh1 | PatchTST | `1.7881e-7` | `7.1526e-7` |

These residuals are implementation-level floating-point effects; the exact
permutation theorem is algebraic.

## 2. Public side-information phase diagram

### 2.1 Exact combinatorial regimes

- Workflow run: `29411342844`
- Artifact: `ett-channel-side-information`
- Artifact id: `8341311121`
- Artifact digest:
  `sha256:a8be8c7388c2e3d315e41018dfb9a44b63918004eebfba9cf076366bb101e408`
- Quality gate: passed

For seven distinct private channels:

| side information | residual orbit | exact labeled-order ceiling |
|---|---:|---:|
| ordered semantic labels private | `5040` | `1/5040` |
| four public channel families of sizes `2,2,2,1` | `8` | `1/8` |
| all semantic channel identities public | `1` | `1` |

Recovery modulo the residual permutation has information-theoretic ceiling one;
this does not imply that a canonical orbit representative is computationally easy
to recover.

### 2.2 Empirical public-calibration attacker

- Workflow run: `29411342754`
- Artifact: `ett-channel-side-information-publication`
- Artifact id: `8341378361`
- Artifact digest:
  `sha256:f9844a51b25395af89764e14e1d152fdbb998b98a818ce82605523a034fa5ecc`
- Quality gate: passed

For each of ETTm1, ETTm2 and ETTh1, the experiment uses 20 public labeled
calibration windows, 20 disjoint private evaluation windows, and five independent
channel permutations per evaluation window: 100 trials per dataset and 300 trials
overall. No trial failed.

| dataset | exact semantic order | mean channel accuracy | 95% bootstrap interval |
|---|---:|---:|---:|
| ETTm1 | `0/100` | `0.2643` | `[0.2285, 0.2972]` |
| ETTm2 | `5/100` | `0.4214` | `[0.3771, 0.4657]` |
| ETTh1 | `0/100` | `0.2214` | `[0.1886, 0.2543]` |

The no-side-information exact-order ceiling is approximately `1/5040` for ETTm1
and ETTh1. ETTm2 has a slightly larger mean ceiling (`0.0002282`) because some
windows contain duplicate complete channel signatures.

This experiment demonstrates that the theorem's privacy guarantee is conditional:
public labeled calibration data can recover part of the channel identity. The
declared diagonal-Gaussian temporal matcher is not Bayes-optimal, so its failures
must not be interpreted as universal semantic privacy.

## 3. Paired learned-initializer confirmation

- Workflow run: `29411342850`
- Artifact: `gifteval-patchtst-paired-learned-publication`
- Artifact id: `8341319832`
- Artifact digest:
  `sha256:bf2aa66e7bbba4f8363cc5309d04e7bfd2525369bd1ba1eecc5b32019894da9d`
- Quality gate: passed

The revision-pinned GIFT-Eval PatchTST study uses 20 confirmatory records disjoint
from the auxiliary initializer-calibration split and three restarts per method and
record. All attempts completed.

The learned quantile initializer improves MSE on `12/20` paired records, but the
paired mean learned-minus-random MSE is `+0.086923` with 95% bootstrap interval
`[0.001632, 0.205662]`. The median difference is close to zero. Thus the learned
initializer is not uniformly stronger: occasional large failures make its mean
performance significantly worse. This negative outcome remains in the evidence
ledger and prevents selective reporting of only improved records.

## 4. Quantized release closure: retained failure and strict rerun

### 4.1 Retained float32 boundary failure

- Workflow run: `29411342920`
- Artifact: `ett-cross-dataset-channel-permutation`
- Artifact id: `8341397991`
- Artifact digest:
  `sha256:33a931c346bbd6cf2fd069e2604d56df49cfb8382b942cb6a78c416d808b8eff`
- Quality gate: failed

The fibre theorem passed in all four ETTm2/ETTh1 × iTransformer/PatchTST cells.
For ETTh1 × iTransformer, however, one of 20 float32 executions crossed a fixed
8-bit quantizer rounding boundary at scale `1e-3`. The raw gradients differed only
by numerical reduction noise, but the discontinuous quantizer amplified that
residual to one output step (`0.001`). The failure is retained as evidence.

### 4.2 Predeclared float64 numerical witness

- Workflow run: `29412797449`
- Artifact: `ett-cross-dataset-channel-permutation`
- Artifact id: `8341885216`
- Artifact digest:
  `sha256:431ae7fc312751642345564aeccb5b33794a016c3e78e403a8d3110b0a489049`
- Quality gate: passed

The rerun keeps the same 8-bit quantizer and the same `1e-3` scale. It does not
weaken the release mechanism or relax the acceptance threshold. Instead, the fixed
trained model and release audit are evaluated under a predeclared float64 contract
with tolerance `1e-10`.

All four cells pass. Each release variant passes on `20/20` windows:

- full exact gradient;
- global clipping at norm `0.5`;
- fixed 8-bit quantization at scale `1e-3`;
- independent Gaussian noise with standard deviation `0.01` under shared audit
  randomness;
- first-parameter-only visibility.

The maximum raw-gradient discrepancy over all cells is `1.7764e-15`; the fixed
quantized releases are bit-identical (`0.0` maximum discrepancy). This confirms
that the earlier failure was a finite-precision witness failure rather than a
counterexample to deterministic postprocessing closure.

## 5. Claim boundary after this update

Supported:

- exact channel-permutation non-identifiability for anonymous-channel iTransformer
  and shared-head channel-independent PatchTST;
- replication over ETTm1, ETTm2 and ETTh1, including larger models and long
  lookbacks;
- exact Bayes ceilings for labeled channel order;
- closure under the declared clipping, quantization, Gaussian-noise and partial
  release channels;
- an explicit side-information phase diagram and a 300-trial calibration attack;
- imperfect but nontrivial GIFT-Eval PatchTST reconstruction and paired negative
  initializer evidence.

Not supported:

- practical or asymptotic end-to-end quantum advantage;
- a coherent compiler for full attention, Softmax, normalization and RevIN;
- universal semantic-channel privacy in the presence of arbitrary external side
  information;
- exact reconstruction of all PatchTST training records;
- hardware speedup claims based on GitHub-hosted runner time.

The current top-tier route is a focused negative theorem paper. The next empirical
requirements are an additional immutable external dataset, a frozen publication
environment, and independent theorem/code review. Positive quantum-advantage claims
remain excluded unless a common-unit `C_Q < C_C` region is established against the
strongest matched classical pipeline.
