# Multistep Training-Transcript Non-Identifiability

## 1. From one gradient to an entire local-training release

The channel-permutation theorem gives, for every model parameter value,

\[
\mathcal L(\theta;XP,YP)=\mathcal L(\theta;X,Y)
\tag{1}
\]

whenever the forecasting model is channel-permutation equivariant and the loss is
invariant under simultaneous permutation of prediction and target channels.
Differentiating (1) yields identical complete gradients. Because the equality holds
for **all** parameter values rather than only at one checkpoint, it also makes the
entire deterministic optimizer transcript identical.

This matters for federated and distributed training, where the server may receive a
model delta after several local steps rather than one raw gradient.

## 2. Deterministic optimizer theorem

Let the optimizer state at step `t` be `s_t` and let a deterministic first-order
transition be

\[
(\theta_{t+1},s_{t+1})
=
\Phi_t(\theta_t,s_t,\nabla_\theta\mathcal L(\theta_t;D)).
\tag{2}
\]

The transition may implement SGD, momentum, Adam, AdamW, clipping, deterministic
quantization, or any composition of public deterministic operations.

### Theorem 1 — complete optimizer trajectories coincide

Suppose datasets `D` and `D'` satisfy

\[
\mathcal L(\theta;D)=\mathcal L(\theta;D')
\quad\text{for every }\theta.
\tag{3}
\]

If both executions start from the same parameters and optimizer state and use the
same deterministic transition (2), then for every step:

\[
\theta_t(D)=\theta_t(D'),
\qquad
s_t(D)=s_t(D'),
\tag{4}
\]

and therefore every released gradient, checkpoint, optimizer-state tensor, and
model delta is identical.

### Proof

The initial states coincide by assumption. If the states coincide at step `t`, (3)
implies that the gradients at that shared parameter value coincide. Applying the
same deterministic transition to identical arguments gives identical states at
step `t+1`. Induction proves (4). `□`

For simultaneous channel permutations of anonymous-channel PatchTST or
iTransformer training data, condition (3) follows from the channel-permutation loss
identity. Hence multiple local training steps do not recover the missing labeled
channel ordering.

## 3. Randomized and noisy releases

If optimizer or release randomness is independent of the private channel labels,
the two executions can be coupled with the same random seed. Their conditional
transcripts are then identical for every seed, so their transcript distributions
are identical.

Consequently, the indistinguishability survives:

- stochastic rounding;
- additive noise drawn independently of channel labels;
- random sparsification masks;
- randomized quantization;
- subsampling schedules coupled across the two orbit members;
- any stochastic post-processing applied to an already identical gradient or model
  delta.

This is a direct data-processing consequence. Noise can hide additional
information, but it cannot reveal which member of an already indistinguishable
channel-permutation orbit generated the release.

## 4. Exact recovery ceiling

If the private batch has `C` distinct labeled channel signatures, the orbit has
`C!` members. Under a uniform prior on that orbit, every estimator observing an
arbitrary deterministic or randomized transcript described above satisfies

\[
P_{\mathrm{exact\ ordered}}\le\frac{1}{C!}.
\tag{5}
\]

With signature multiplicities `m_1,...,m_r`, the ceiling becomes

\[
P_{\mathrm{exact\ ordered}}
\le
\frac{\prod_jm_j!}{C!}.
\tag{6}
\]

The bound applies equally to classical and quantum reconstruction because every
member induces the same released transcript and therefore the same coherent oracle
constructed from that transcript.

## 5. Executable certificate

`qrecon.theory.channel_permutation_training` provides:

- `channel_permutation_training_transcript_witness`;
- `ChannelPermutationTrainingTranscriptWitness`;
- stepwise loss, full-gradient, parameter, and optimizer-state differences;
- final model-delta difference;
- support for SGD, momentum SGD, Adam, and AdamW.

Tests cover anonymous-channel iTransformer, shared-head PatchTST, all four optimizers,
and a channel-specific PatchTST-head control that must break the symmetry.

The certificate uses deterministic full-batch MSE. Stochastic layers must be
disabled or their randomness explicitly coupled. Small numerical residuals are
allowed because a channel permutation can change floating-point reduction order.

## 6. Security interpretation

The theorem closes a common escape route in gradient-reconstruction claims. Once a
one-step collision is identified through a loss identity that holds for every
parameter value, releasing more local steps does not necessarily make the original
labeled object identifiable. In the anonymous-channel setting, all of the following
remain in the same observation fibre:

- raw complete gradients;
- clipped or quantized gradients;
- momentum or Adam state updates;
- multi-step local SGD trajectories;
- final FedAvg-style model deltas;
- deterministic or label-independent randomized post-processing.

A paper must therefore declare whether semantic channel labels are public and
whether channel order is part of the recovery target. Reporting only numerical
reconstruction error after choosing one arbitrary orbit member can conceal an
information-theoretic impossibility.

## 7. Claim boundary

The theorem does not apply when the update rule receives private channel identity as
an additional input or when the model/loss contains channel-indexed parameters,
embeddings, heads, weights, or metadata. Those mechanisms deliberately break the
symmetry and are included as controls in the real ETTm1 ablation. It also does not
prove that the unordered set of channel trajectories is unrecoverable.
