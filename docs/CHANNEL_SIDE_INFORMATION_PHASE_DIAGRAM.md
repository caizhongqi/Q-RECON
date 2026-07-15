# Channel Side Information and Recovery-Target Phase Diagram

## 1. Purpose

Channel-permutation non-identifiability is a statement about a declared observation
channel, private object, prior, and recovery target. It must not be transferred
unchanged between threat models that reveal different semantic metadata.

This document makes the side-information boundary exact. Public channel metadata is
modeled as a partition of the channel positions. Only permutations preserving the
public metadata remain compatible with the attacker's knowledge.

## 2. Residual subgroup

Let the private channel signatures be

\[
z_1,\ldots,z_C,
\]

where each signature includes every ordered history and forecast-target coordinate
whose channel identity belongs to the private recovery target. Let

\[
s_1,\ldots,s_C
\]

be the public side-information labels. The admissible permutation subgroup is

\[
H_s=\{P\in S_C:s_{P(c)}=s_c\ \text{for all }c\}.
\]

If a public-label class `a` contains `n_a` channels and its distinct private
signatures have multiplicities

\[
m_{a,1},\ldots,m_{a,r_a},
\]

then the number of distinct private ordered objects remaining in the observation
fibre is

\[
|\mathcal O_s|
=
\prod_a
\frac{n_a!}{\prod_jm_{a,j}!}.
\tag{1}
\]

### Theorem — side-information-restricted Bayes ceiling

Assume the released gradient observation is invariant under simultaneous channel
permutations and the prior is uniform on the residual orbit in (1). Then every
classical estimator and every coherent quantum algorithm using only that observation
and the declared public side information satisfies

\[
P_{\mathrm{exact\ ordered}}^*
=
\frac{1}{|\mathcal O_s|}.
\tag{2}
\]

#### Proof

Public labels remove permutations outside `H_s`. All elements of the residual
`H_s`-orbit still induce the same gradient observation. Identical private signatures
inside one public class describe the same ordered private object after exchange, so
orbit-stabilizer counting gives (1). Under the uniform residual-orbit prior, every
observation has the same likelihood for each distinct member; the Bayes-optimal exact
ordered guess therefore succeeds with probability `1/|O_s|`. The coherent oracle is
identical on the same residual orbit, so a quantum algorithm has the same posterior
ceiling. `□`

## 3. Recovery modulo permutation

If the declared recovery target treats all residual orbit members as equivalent, the
entire orbit is one target class. The information-theoretic ceiling for recovering
that equivalence class is then one.

This statement must not be misread as an algorithmic result: it does not prove that an
attacker can construct any orbit representative from the gradient. It only says that
channel ordering is no longer part of the success criterion.

## 4. ETT seven-channel phase diagram

The validated ETTm1 and ETTh1 windows have seven distinct complete private channel
signatures, while two ETTm2 windows contain one duplicated pair. For the generic
seven-distinct-channel case:

| side-information regime | public groups | residual orbit | exact ordered ceiling |
|---|---|---:|---:|
| ordered labels private | all seven channels anonymous | `7! = 5040` | `1/5040` |
| channel family public | pairs `(HUFL,HULL)`, `(MUFL,MULL)`, `(LUFL,LULL)`, plus `OT` | `2!·2!·2! = 8` | `1/8` |
| full semantic labels public | seven singleton groups | `1` | `1` from this ambiguity alone |
| target modulo residual permutation | orbit collapsed to one target class | `1` class | `1` for the quotient target |

The family-label row is an interpretable intermediate regime, not a claim that a
specific deployment publishes exactly those categories.

## 5. Executable mapping

`qrecon.theory.channel_side_information` provides:

- `ChannelSideInformationBound`;
- `channel_side_information_bound`;
- `channel_side_information_phase_diagram`.

`examples/ett_channel_side_information_phase_diagram.py` produces a machine-readable
ETT report with a content hash. Tests cover fully private labels, family labels,
unique public labels, duplicate private signatures, quotient targets, and malformed
contracts.

## 6. Claim boundary

The phase diagram isolates the uncertainty caused specifically by channel-permutation
symmetry. It does not rule out other fibres after full semantic labels become public,
and it does not model distributional side information that correlates a numerical
trajectory with a semantic label. Such information must be included as an explicit
additional observation or prior before recomputing the Bayes risk.

The defensible result is conditional but exact:

> For every declared public-label partition, the residual permutation orbit and its
> classical/quantum exact-order recovery ceiling are given by (1)–(2).
