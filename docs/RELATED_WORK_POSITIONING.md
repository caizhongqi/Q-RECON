# Related-Work Positioning for the Q-RECON Submission

## 1. Purpose

This document states the intended novelty boundary before paper writing. It prevents
Q-RECON from claiming contributions already established by modern gradient inversion
or Transformer-privacy work.

The primary references below are cited by stable title and arXiv identifier. A final
submission must replace this internal summary with a complete, venue-formatted
bibliography and verify the latest published versions.

## 2. Time-series gradient inversion

### TS-Inverse

**Caspar Meijer et al., “TS-Inverse: A Gradient Inversion Attack Tailored for
Federated Time Series Forecasting Models,” arXiv:2503.20952 (2025).**

TS-Inverse studies time-series reconstruction empirically across multiple forecasting
models and datasets. Its main attack contributions are a learned quantile initializer,
a time-series-specific objective with trend/periodicity terms, and regularization
using learned quantiles. It reports strong sMAPE improvements over prior gradient
inversion baselines.

Q-RECON must therefore **not** claim the following as new:

- that time-series forecasting gradients leak observations or targets;
- that trend and periodicity regularizers can help inversion;
- that learned quantile information can initialize a time-series attack;
- that PatchTST or another forecasting Transformer is vulnerable merely because a
  gradient optimizer achieves a low reconstruction error.

Q-RECON's distinct contribution is instead the information layer:

- exact observation fibres induced by architecture symmetries;
- Bayes-optimal exact-order ceilings;
- classical and coherent quantum indistinguishability;
- closure under deterministic optimizer transcripts and randomized release channels;
- an explicit public-side-information phase diagram.

The repository nevertheless includes a declared TS-Inverse-style objective baseline
and a paired learned-quantile study so the theorem is not evaluated against weak
optimization baselines. The learned initializer is not uniformly beneficial in the
current confirmatory PatchTST study, and that negative result remains in the evidence
ledger.

## 3. Transformer gradient leakage

### Partial Transformer gradients

**Weijun Li, Qiongkai Xu, and Mark Dras, “Seeing the Forest through the Trees: Data
Leakage from Partial Transformer Gradients,” arXiv:2406.00999 (2024).**

This work demonstrates that training data can be reconstructed from a single
Transformer layer or even a small linear subcomponent. It establishes that partial
visibility is often sufficient for leakage.

Q-RECON must therefore not present “a Transformer leaks from partial gradients” as a
new finding. The non-overlapping question is whether different private objects are
**provably indistinguishable even under the complete gradient**. Q-RECON's
channel-permutation theorem supplies such a fibre for anonymous-channel forecasting
Transformers and then proves that taking a fixed visible subset cannot split it.

### Theoretical Transformer leakage

**Chenyang Li et al., “A Theoretical Insight into Attack and Defense of Gradient
Leakage in Transformer,” arXiv:2311.13624 (2023).**

This work studies conditions for Transformer gradient reconstruction and gradient
noise defenses. Q-RECON's theory should not be described as the first theoretical
analysis of Transformer leakage.

The intended distinction is:

- prior theory asks when a private input can be reconstructed from a declared
  Transformer gradient;
- Q-RECON identifies a group orbit of private multivariate input/target pairs that
  induces exactly the same gradient for every parameter value;
- Q-RECON propagates this equality through coherent queries, multi-step optimizer
  state, and general postprocessing/randomized release kernels.

## 4. General gradient reconstruction and defenses

DLG, iDLG, InvG, analytical linear-layer attacks, convolutional analytical attacks,
SMT/branch-and-bound inversion, and gradient-obfuscation studies establish that
optimization quality and privacy are architecture- and release-dependent. Q-RECON
uses these methods as baselines or boundary cases rather than claiming their core
ideas.

In particular, low reconstruction error is not proof of unique recovery. Every
experiment must report the declared recovery equivalence and, when tractable, the
observation fibre or an information-theoretic bound.

## 5. Forecasting backbones

### PatchTST

**Yuqi Nie et al., “A Time Series is Worth 64 Words: Long-term Forecasting with
Transformers,” arXiv:2211.14730, ICLR 2023.**

PatchTST contributes patch tokens and channel independence. Q-RECON uses a
channel-independent shared-head PatchTST as a modern victim and as one architecture
satisfying the permutation-equivariance theorem. The use of PatchTST itself is not a
Q-RECON contribution.

### iTransformer

**Yong Liu et al., “iTransformer: Inverted Transformers Are Effective for Time
Series Forecasting,” arXiv:2310.06625, ICLR 2024 Spotlight.**

iTransformer embeds complete variable histories as tokens and attends across
variables. Q-RECON uses an anonymous-channel version without channel-indexed
parameters. Again, the backbone is prior work; the new result is the training-gradient
fibre induced by its equivariance.

## 6. Q-RECON novelty matrix

| Question | Prior work establishes | Q-RECON target contribution |
|---|---|---|
| Can time-series gradients leak data? | Yes, including tailored TS attacks | Not claimed as new |
| Can partial Transformer gradients leak? | Yes | Not claimed as new |
| Can modern forecasting Transformers be attacked? | Empirically plausible/known | Matched baselines only |
| Can two distinct private forecasting objects have exactly equal full gradients for every parameter value? | Not the central result of the references above | Exact channel-permutation theorem |
| What is the exact labeled-order Bayes ceiling? | Not supplied for this group orbit | Orbit-stabilizer formula and side-information restriction |
| Can coherent quantum queries overcome the ambiguity? | Not addressed in the cited attack work | Identical-oracle impossibility corollary |
| Does the ambiguity survive local training and gradient defenses? | Individual defenses studied empirically/theoretically | Deterministic optimizer and release-kernel closure |
| Do public semantic labels remove the ambiguity? | Threat-model dependent | Exact residual subgroup phase diagram plus empirical calibration study |

## 7. Defensible paper claim

The intended primary claim is:

> In anonymous-channel multivariate forecasting, equivariant modern Transformer
> architectures induce exact training-data permutation fibres. We characterize the
> resulting Bayes limits under explicit side-information regimes and show that the
> ambiguity is shared by classical and coherent quantum attackers, deterministic
> first-order training transcripts, and common gradient-release mechanisms.

The intended empirical contribution is supporting evidence across immutable ETT
variants, two modern architecture families, symmetry-breaking controls, matched
optimization baselines, and explicit side-information studies.

The paper must not claim practical quantum advantage. Its quantum result is an
information-theoretic impossibility statement: an identical coherent oracle cannot
reveal a missing orbit label.
