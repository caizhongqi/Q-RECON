# Channel-Permutation Non-Identifiability in Multivariate Transformers

## 1. Motivation

Multivariate forecasting datasets assign semantic labels to variables: for example,
`HUFL`, `HULL`, and `OT` in ETTm1. A forecasting architecture does not necessarily
use those labels. Channel-independent PatchTST and an iTransformer without
channel-indexed parameters can treat the variable axis as an unordered collection.
This architectural symmetry creates an exact input-level training-gradient fibre.
It is not an optimizer failure and it is not specific to a weak attack.

The result below applies to the **complete model gradient**, including every
attention, feed-forward, projection, positional, normalization-free, and forecasting
head parameter. It therefore constrains classical white-box reconstruction and any
quantum procedure whose observation oracle is built only from that released
gradient.

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

for every parameter value `theta`, input batch `X`, and permutation `P`.
Training uses the mean squared error

\[
\mathcal L(\theta;X,Y)
=
\frac{1}{BHC}\lVert f_\theta(X)-Y\rVert_F^2.
\tag{2}
\]

## 3. Full-gradient invariance theorem

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

### Proof

By equivariance,

\[
f_\theta(XP)-YP
=
(f_\theta(X)-Y)P.
\]

A permutation matrix is orthogonal, so right multiplication preserves the
Frobenius norm:

\[
\lVert(f_\theta(X)-Y)P\rVert_F
=
\lVert f_\theta(X)-Y\rVert_F.
\]

This proves (3) as an identity in the model parameters. Differentiating that
identity proves (4) for every parameter tensor. `□`

The conclusion is stronger than equality of a selected layer or a final-head
statistic: the entire released parameter-gradient tuple is identical.

## 4. Exact observation fibre and Bayes ceiling

For channel `c`, define its complete private signature as the concatenation of all
history and target entries belonging to that labeled channel. Suppose the distinct
signatures have multiplicities

\[
m_1,\ldots,m_r,\qquad \sum_jm_j=C.
\]

Permutations within an identical-signature group do not create a new private
object. The number of distinct labeled training objects in the simultaneous
permutation orbit is therefore

\[
|\mathcal O(X,Y)|
=
\frac{C!}{\prod_{j=1}^r m_j!}.
\tag{5}
\]

Every orbit member has the same complete gradient by Theorem 1. Under a uniform
prior on this orbit, the Bayes-optimal probability of recovering the exact labeled
ordering is at most

\[
P^*_{\mathrm{ordered}}
=
\frac{\prod_jm_j!}{C!}.
\tag{6}
\]

For generic distinct channels, `m_j=1` and the ceiling is `1/C!`. For the seven
ETTm1 variables this is

\[
\frac{1}{7!}=
\frac{1}{5040}\approx1.984\times10^{-4}.
\]

No classical or quantum reconstruction algorithm can exceed this ceiling from the
gradient observation alone. A coherent oracle constructed from the same gradient
channel is identical on every orbit member and cannot restore the missing semantic
channel labels.

## 5. Architectures covered by the repository

The theorem applies only when equivariance (1) holds for every parameter value.
In the current Q-RECON implementations this includes:

- `ITransformer(..., revin=False)`, because complete variable histories are tokens,
  attention and output projection weights are shared, and no channel embedding is
  present;
- `PatchTST(..., revin=False, individual_head=False)`, because channels are folded
  into the batch axis and share the patch projection, encoder, positional embedding,
  and forecasting head.

The theorem does **not** apply unchanged when the model contains:

- channel embeddings or variable identifiers;
- channel-specific forecasting heads;
- learned per-channel affine RevIN parameters;
- externally supplied semantic metadata;
- a loss that weights channels differently using their labels.

These mechanisms can break the symmetry. They do not invalidate the theorem; they
change the observation model and must be analyzed separately.

## 6. Executable certificate

`qrecon.theory.channel_permutation` provides:

- `channel_permutation_orbit_size`;
- `channel_permutation_fibre_bound`;
- `tensor_channel_permutation_fibre_bound`;
- `apply_channel_permutation`;
- `channel_permutation_gradient_witness`.

The witness computes on a concrete model and batch:

- prediction equivariance error;
- loss difference;
- maximum absolute difference over every parameter-gradient tensor;
- relative full-gradient L2 difference;
- private input/target displacement;
- orbit size and exact labeled-recovery ceiling.

Unit tests cover both iTransformer and channel-independent PatchTST with nonlinear
attention and feed-forward blocks. The real-data experiment
`examples/ettm1_itransformer_channel_permutation.py` evaluates twenty revision-pinned
ETTm1 windows and fails its publication quality gate unless every window has a
nontrivial private orbit and full-gradient agreement within the declared numerical
tolerance.

## 7. Interpretation for Q-RECON

This result extends Q-RECON's negative theory beyond linear regression and additive
two-sum structure. It gives a modern nonlinear Transformer setting in which exact
labeled reconstruction is information-theoretically impossible even when the
attacker receives all parameter gradients with infinite numerical precision.

The result separates three recovery targets that must not be conflated:

1. **labeled-channel recovery** — recover the exact semantic variable ordering;
2. **orbit recovery** — recover the unordered set of channel histories and targets;
3. **numerical consistency** — find any candidate producing the released gradient.

A gradient-matching optimizer may succeed at (3) while necessarily failing at (1).
Paper experiments must therefore report the declared target equivalence and the
orbit-aware Bayes ceiling before interpreting reconstruction error.

## 8. Claim boundary

The theorem is an impossibility result for anonymous-channel equivariant models.
It does not show that all PatchTST or iTransformer deployments hide channel labels,
and it does not imply that the unordered orbit representative is unrecoverable. A
top-tier claim should combine this theorem with:

- real-data verification over multiple architectures and channel counts;
- explicit symmetry-breaking ablations;
- partial, clipped, quantized, and noisy gradient releases;
- orbit-aware attack metrics;
- a statement of whether semantic variable labels are public side information.
