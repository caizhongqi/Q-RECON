# Channel-Permutation Non-Identifiability in Multivariate Transformers

## 1. Motivation

Multivariate forecasting datasets assign semantic labels to variables: for example,
`HUFL`, `HULL`, and `OT` in the ETT family. A forecasting architecture does not
necessarily use those labels. Shared-head channel-independent PatchTST and an
iTransformer without channel-indexed parameters treat the variable axis as an
anonymous collection. This architectural symmetry creates an exact training-data
observation fibre.

The result is not an optimizer failure and it is not specific to a weak attack. It
applies to:

- the complete model gradient;
- deterministic first-order optimizer transcripts and final model updates;
- clipped, quantized, partially visible, and independently randomized gradient
  releases;
- classical estimators and coherent quantum query algorithms whose oracle is built
  only from those observations.

The private object in this document contains both ordered input histories and ordered
forecast targets. Public semantic channel labels or a recovery target defined only
modulo permutation are different threat models.

## 2. Setting

Let

\[
X\in\mathbb R^{B\times L\times C},\qquad
Y\in\mathbb R^{B\times H\times C}
\]

be a multivariate forecasting training batch with `C` labeled channels. A channel
permutation is represented by a permutation matrix
\(P\in\{0,1\}^{C\times C}\), acting on the last tensor axis.

Let a differentiable model

\[
f_\theta:\mathbb R^{B\times L\times C}
\longrightarrow \mathbb R^{B\times H\times C}
\]

be **channel-permutation equivariant** when

\[
f_\theta(XP)=f_\theta(X)P
\tag{1}
\]

for every parameter value `theta`, input batch `X`, and permutation `P`. Training
uses the mean squared error

\[
\mathcal L(\theta;X,Y)
=
\frac{1}{BHC}\lVert f_\theta(X)-Y\rVert_F^2.
\tag{2}
\]

## 3. Full-gradient invariance

### Theorem 1 — simultaneous channel permutations have identical full gradients

Under (1) and (2), for every channel permutation `P`,

\[
\mathcal L(\theta;XP,YP)=\mathcal L(\theta;X,Y)
\tag{3}
\]

for every `theta`, and consequently

\[
\nabla_\theta\mathcal L(\theta;XP,YP)
=
\nabla_\theta\mathcal L(\theta;X,Y).
\tag{4}
\]

#### Proof

By equivariance,

\[
f_\theta(XP)-YP=(f_\theta(X)-Y)P.
\]

A permutation matrix is orthogonal, so right multiplication preserves the
Frobenius norm. This proves (3) as an identity in the model parameters.
Differentiating that identity proves (4) for every parameter tensor. `□`

The conclusion concerns the complete parameter-gradient tuple, including attention,
feed-forward, projection, positional, normalization, and forecasting-head
parameters. Checking only one selected layer is not sufficient evidence for this
theorem.

## 4. Exact fibre and Bayes ceiling

For channel `c`, define its complete private signature as the concatenation of all
history and target entries belonging to that labeled channel. Suppose the distinct
signatures have multiplicities

\[
m_1,\ldots,m_r,\qquad \sum_jm_j=C.
\]

Permutations within an identical-signature group do not create a new private object.
The number of distinct labeled training objects in the orbit is

\[
|\mathcal O(X,Y)|
=
\frac{C!}{\prod_{j=1}^r m_j!}.
\tag{5}
\]

Every orbit member has the same complete gradient by Theorem 1. Under a uniform
prior on this orbit, the Bayes-optimal probability of recovering the exact labeled
ordering is

\[
P^*_{\mathrm{ordered}}
=
\frac{\prod_jm_j!}{C!}.
\tag{6}
\]

For seven distinct channels this is

\[
\frac{1}{7!}=\frac{1}{5040}\approx1.984\times10^{-4}.
\]

For a nonuniform prior, the corresponding optimum is the largest posterior mass of
an orbit member. External semantic side information can therefore change the Bayes
risk even though it does not change the equality of the released gradients.

## 5. Classical and quantum indistinguishability

### Corollary 1 — coherent access cannot recover the missing channel order

Let an observation oracle be any deterministic function of the full gradient, or a
clean coherent implementation of that function. Because (4) holds for every
parameter query, all orbit members induce the same classical oracle function and the
same unitary oracle. Any adaptive classical transcript and any quantum algorithm's
final density operator are therefore identical across the orbit.

Consequently, no number of classical or coherent quantum queries can exceed the
Bayes ceiling in (6) using this observation channel alone. The statement is
information-theoretic; it does not rely on computational hardness or a conjectured
quantum lower bound.

## 6. Multi-step training transcripts

### Theorem 2 — deterministic first-order optimization preserves the fibre

Let the optimizer state be `s_t` and suppose the update is deterministic:

\[
(\theta_{t+1},s_{t+1})
=F_t(\theta_t,s_t,g_t),
\qquad
g_t=\nabla_\theta\mathcal L(\theta_t;X,Y).
\tag{7}
\]

Start two executions from the same parameters and optimizer state, using `(X,Y)`
and `(XP,YP)`. By Theorem 1 their gradients are equal at `t=0`, so (7) gives equal
next states. Induction proves equality of every later gradient, optimizer state,
checkpoint, and final model delta. `□`

This covers deterministic SGD, momentum SGD, Adam, and AdamW. Stochastic data order,
dropout, or other randomness must either be public/common-coupled or modeled as a
randomized release channel. The executable certificate uses float64 auditing because
float32 reduction residuals near zero can be amplified by coordinate-wise adaptive
normalization even though the exact mathematical trajectories coincide.

## 7. Release-channel closure

### Theorem 3 — postprocessing and independent randomization cannot split the fibre

Let `G(X,Y)` be the exact gradient observation and suppose

\[
G(XP,YP)=G(X,Y).
\]

For any deterministic map `h`,

\[
h(G(XP,YP))=h(G(X,Y)).
\tag{8}
\]

For any Markov kernel `K` whose conditional law depends on the private object only
through `G`,

\[
K(\cdot\mid G(XP,YP))=K(\cdot\mid G(X,Y)).
\tag{9}
\]

Thus global clipping, fixed or adaptive quantization, a fixed subset of visible
parameter tensors, and additive noise independent of the orbit member cannot improve
exact channel-order recovery. The same conclusion holds for repeated or adaptive
releases whose public history is generated from previously indistinguishable
observations. `□`

The implementation couples randomized mechanisms with the same audit seed to obtain
a pathwise numerical witness. The theorem itself is equality in distribution; it
does not assume the attacker observes the noise realization.

## 8. Architectures covered

The theorem applies only when (1) holds for every parameter value. In Q-RECON this
includes:

- `ITransformer(..., revin=False)`;
- `ITransformer(..., revin=True, revin_affine=False)`;
- `PatchTST(..., individual_head=False, revin=False)`;
- `PatchTST(..., individual_head=False, revin=True, revin_affine=False)`.

Parameter-free RevIN normalizes every channel by the same declared operation and
preserves permutation equivariance. The theorem does **not** apply unchanged to:

- channel embeddings or variable identifiers;
- channel-specific forecasting heads;
- learned per-channel affine RevIN parameters;
- externally supplied semantic metadata;
- a loss that weights channels differently using their labels.

These mechanisms are explicit symmetry-breaking controls, not exceptions to the
proof.

## 9. Executable certificates

The repository provides:

- `qrecon.theory.channel_permutation` for orbit sizes, tensor fibres, permutations,
  and one-step full-gradient witnesses;
- `qrecon.theory.channel_permutation_training` for deterministic optimizer
  transcripts, optimizer states, checkpoints, and final model deltas;
- `qrecon.benchmarks.channel_permutation_fibre` for generator-complete real-data
  fibre studies;
- `qrecon.benchmarks.channel_permutation_release` for clipping, quantization,
  Gaussian noise, partial visibility, and combined release audits.

Adjacent transpositions are checked because they generate the full symmetric group
`S_C`. Passing every generator proves that the complete simultaneous permutation
orbit is contained in one observation fibre, subject to the declared numerical
tolerance.

## 10. Validated empirical evidence

### ETTm1

GitHub Actions run `29409084808`, artifact
`ettm1-channel-permutation-publication`, validates:

- 20 immutable ETTm1 windows;
- generator-complete iTransformer full-gradient fibres;
- shared-head PatchTST as a second equivariant architecture;
- a channel-specific PatchTST head as a symmetry-breaking control;
- three-step AdamW gradient, optimizer-state, checkpoint, and model-delta equality;
- full, clipped, fixed-quantized, Gaussian-noisy, partial, and combined releases.

All publication gates pass. For the anonymous iTransformer, all 20 AdamW transcripts
are certified identical; the maximum observed full-gradient discrepancy is below
`3e-15` in the float64 audit, and the uniform exact labeled-order ceiling is
`1/5040`. The channel-specific control breaks the transcript on all 20 windows.

### ETTm2 and ETTh1

GitHub Actions run `29409660763`, artifact
`ett-cross-dataset-channel-permutation`, validates four cells:

- ETTm2 × iTransformer;
- ETTm2 × shared-head PatchTST;
- ETTh1 × iTransformer;
- ETTh1 × shared-head PatchTST.

Each cell contains 20 immutable real windows. Every fibre and release quality gate
passes. ETTh1 has orbit size `5040` on all 20 windows. ETTm2 has orbit size `5040`
on 18 windows and `2520` on two windows because one pair of complete private channel
signatures is duplicated. Every declared clipping, quantization, noise, and partial
visibility check remains inside the same observation fibre.

Together with ETTm1, the validated main matrix covers 120 modern-model real-data
windows across three datasets and two architecture families, plus explicit
symmetry-breaking controls and multi-step training/release studies.

## 11. Interpretation for reconstruction research

The theorem separates three recovery targets:

1. **exact labeled-channel recovery** — recover the original semantic ordering;
2. **orbit recovery** — recover an unordered representative modulo channel
   permutation;
3. **numerical consistency** — find any candidate producing the released gradient.

A gradient-matching attack can succeed at (3), and potentially at (2), while exact
success at (1) remains bounded by (6). Papers must therefore state the target
equivalence before converting MSE or gradient matching into a privacy claim.

## 12. Claim boundary

This is an impossibility theorem for anonymous-channel equivariant forecasting
models when the ordered histories and ordered future targets are private. It does not
show that all PatchTST or iTransformer deployments hide channel labels, and it does
not prove that an unordered orbit representative is unrecoverable. Public semantic
channel labels, channel-specific parameters, known ordered targets, or strong
external distributional priors can reduce or eliminate the ambiguity.

The defensible top-tier claim is therefore narrower and stronger than a generic
attack claim:

> For a declared anonymous-channel threat model, modern nonlinear forecasting
> Transformers have exact full-gradient training-data fibres whose labeled-order
> ambiguity survives coherent quantum queries, deterministic multi-step optimizers,
> clipping, quantization, independent noise, and partial gradient visibility.

Any broader statement must introduce and analyze the additional side-information
channel explicitly.
