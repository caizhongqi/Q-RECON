# Focused Top-Tier Paper Blueprint

## Working title

**When Gradients Forget Channel Identity: Exact Reconstruction Limits for Modern
Time-Series Transformers**

Alternative security-facing title:

**Gradient Leakage Does Not Identify Semantic Channel Order in Anonymous-Channel
Forecasting Transformers**

## One-sentence thesis

For channel-anonymous, permutation-equivariant forecasting Transformers trained with
a channel-symmetric loss, the complete gradient, deterministic optimizer transcript,
and declared postprocessed releases identify a private multivariate training object
only up to a side-information-conditioned channel orbit; no classical or coherent
quantum reconstruction algorithm can recover the original semantic order above the
corresponding Bayes ceiling.

## Abstract draft

Gradient inversion attacks show that shared model updates can reveal private training
data, including language and time-series records. Existing evaluations usually ask
whether a stronger optimizer can produce a numerically similar reconstruction. We ask
a prior question: does the released gradient uniquely determine the labeled private
object at all? We study multivariate forecasting architectures whose parameters are
shared across anonymous channels, including iTransformer and channel-independent
shared-head PatchTST. We prove that simultaneously permuting private input histories
and forecast targets leaves the loss and every parameter gradient unchanged. The
result induces an exact orbit fibre whose size is computable from duplicate channel
signatures; under a uniform orbit prior, exact semantic-order recovery is bounded by
the reciprocal orbit size for both classical and coherent quantum attackers. We show
that the fibre persists through deterministic SGD, momentum, Adam and AdamW
transcripts and through clipping, quantization, data-independent Gaussian noise,
partial gradient visibility and repeated release channels. We then derive a
side-information phase diagram: public channel metadata restricts the admissible
permutation subgroup and yields an exact residual Bayes ceiling. On immutable ETTm1,
ETTm2 and ETTh1 data, we validate the theorem for iTransformer and shared-head
PatchTST, include channel-specific symmetry-breaking controls, and retain all failed
and numerically unstable checks. A separate GIFT-Eval PatchTST study shows that modern
optimization attacks recover broad waveform structure but achieve zero strict
bitwise or `1e-2` whole-record recovery on the confirmatory split. Our results clarify
which privacy claims follow from architecture-induced information loss, which depend
on external semantic side information, and why improved classical or quantum search
cannot recover distinctions absent from the observation channel.

## Primary contributions

### C1 — exact modern-model gradient fibre theorem

For every parameter value of a declared channel-anonymous equivariant forecaster,

\[
f_\theta(XP)=f_\theta(X)P,
\]

and a channel-symmetric loss satisfies

\[
L(\theta;XP,YP)=L(\theta;X,Y).
\]

Therefore

\[
\nabla_\theta L(\theta;XP,YP)
=
\nabla_\theta L(\theta;X,Y)
\]

for every channel permutation `P`.

### C2 — exact orbit and side-information Bayes ceilings

For private signatures with multiplicities `m_j`, the no-side-information orbit has
size

\[
C!/\prod_j m_j!.
\]

For a public-label partition with class-specific multiplicities `m_{a,j}`, the
residual orbit is

\[
\prod_a n_a!/\prod_jm_{a,j}!,
\]

and exact labeled-order recovery has uniform-prior ceiling equal to its reciprocal.
Recovery modulo the residual orbit is a different target and must be reported
separately.

### C3 — closure under realistic observation transformations

Prove and audit closure under:

- deterministic optimizer transcripts and model deltas;
- deterministic clipping, quantization and parameter projection;
- data-independent randomized release kernels such as additive Gaussian noise;
- repeated conditionally independent releases;
- adaptive classical queries and coherent quantum access to the same fibre-constant
  oracle.

### C4 — real-data, modern-architecture validation with controls

Publication evidence must include:

- ETTm1, ETTm2 and ETTh1 with immutable source hashes;
- iTransformer and shared-head channel-independent PatchTST;
- adjacent generators of the full symmetric group;
- multi-step optimizer studies;
- clipping, quantization, noise and partial-visibility release studies;
- channel-specific heads or affine per-channel RevIN as symmetry-breaking controls;
- a long-context, larger-model stress matrix;
- a public side-information calibration study;
- raw artifacts, failures and numerical tolerances.

### C5 — matched reconstruction evidence, not an exaggerated attack claim

Use GIFT-Eval + PatchTST to report DLG, cosine/InvG, Q-RECON hybrid,
TS-Inverse-style objectives and defenses. Distinguish:

- bitwise exact whole-record recovery;
- every value within named absolute tolerances;
- relative-L2 thresholds;
- MSE, correlation and sMAPE;
- selected-best-restart statistics and all-attempt completion rates.

The current confirmatory result supports nontrivial waveform reconstruction but not
strict exact raw-record recovery.

## Paper structure

1. **Introduction.** Gradient inversion normally optimizes before checking
   identifiability; semantic channel order is a concrete missing-information case.
2. **Threat model and recovery targets.** Ordered labels private/public, public
   calibration, quotient recovery, classical and coherent access.
3. **Permutation fibre theorem.** Loss identity, gradient equality, orbit formula,
   coherent indistinguishability.
4. **Optimizer and release closure.** Deterministic and randomized postprocessing.
5. **Side-information phase diagram.** Residual subgroup, analytical ETT examples,
   empirical calibration attacker.
6. **Experimental protocol.** Immutable datasets, models, controls, tolerances,
   failures, statistics.
7. **Results.** Fibre validation, larger-model stress, training/release closure,
   side information, reconstruction attacks.
8. **Related work.** Gradient inversion, Transformer leakage, TS-Inverse, equivariant
   networks and privacy.
9. **Limitations and deployment consequences.** Public labels, architecture changes,
   unordered target, distributional priors, no positive quantum speedup.
10. **Conclusion.** Gradient leakage can be severe while still failing to identify a
    declared semantic distinction.

## Main tables and figures

### Table 1 — theorem applicability matrix

Rows: iTransformer, shared-head PatchTST, individual-head PatchTST, affine-RevIN,
channel embeddings. Columns: equivariant, exact fibre, control outcome.

### Table 2 — real fibre validation

Dataset × architecture × context/horizon × model size × windows × maximum gradient
error × orbit distribution × exact-order ceiling.

### Table 3 — optimizer and release closure

SGD, momentum, Adam, AdamW; full, clipped, int8, noisy, partial and repeated releases.

### Figure 1 — orbit and side-information phase diagram

`5040 -> 8 -> 1` for private labels, family labels and full semantic labels on the
seven-channel ETT example, with a separate quotient-target branch.

### Figure 2 — strict reconstruction success curves

Whole-record success versus absolute tolerance and relative-L2 threshold for matched
PatchTST attacks and defenses. Never label an arbitrary tolerance as “exact.”

### Figure 3 — empirical public-calibration side information

Exact semantic assignment and per-channel accuracy across three ETT datasets,
including assignment-margin distributions.

## Claims excluded from the paper

- practical or asymptotic quantum speedup;
- a coherent compiler for full PatchTST/iTransformer attention stacks;
- universal privacy from gradient clipping, noise or quantization;
- exact recovery of all PatchTST raw records;
- privacy when semantic channel labels or identifying metadata are public;
- novelty claims based only on adding PatchTST/iTransformer implementations;
- GitHub-hosted wall-clock timings as hardware-comparable performance.

## Submission readiness rule

The draft is ready for external top-tier review only when:

1. every principal theorem has a checked proof and executable witness;
2. all declared publication workflows pass at one immutable commit;
3. the side-information and large-model stress artifacts are archived;
4. strict success metrics replace ambiguous “exact tolerance” terminology;
5. the related-work theorem comparison is reviewed against current literature;
6. an independent reviewer attempts to falsify the theorem, implementation and
   threat-model assumptions;
7. the paper and artifact cite the same hashes, sample indices, tolerances and
   failure counts.
