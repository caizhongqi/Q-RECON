# Channel-Permutation Fibres in Modern Multivariate Forecasters

## 1. Setting

Let a multivariate forecaster

\[
F_\theta:\mathbb R^{T\times C}\to\mathbb R^{H\times C}
\]

share all channel-processing parameters and contain no learned parameter tied to
an absolute channel identity. Let \(P\) be a permutation matrix acting on the
channel axis. Assume

\[
F_\theta(xP)=F_\theta(x)P.
\]

For mean squared forecasting loss,

\[
\ell(\widehat yP,yP)=\ell(\widehat y,y).
\]

The private training object in this section is the ordered pair \((x,y)\), so
channel labels and ordering are part of the secret.

## 2. Full-gradient invariance theorem

For every channel permutation \(P\),

\[
\begin{aligned}
\mathcal O_\theta(xP,yP)
&=\nabla_\theta\ell(F_\theta(xP),yP)\\
&=\nabla_\theta\ell(F_\theta(x)P,yP)\\
&=\nabla_\theta\ell(F_\theta(x),y)\\
&=\mathcal O_\theta(x,y).
\end{aligned}
\]

Thus the complete parameter-gradient observation fibre contains the simultaneous
channel-permutation orbit

\[
\{(xP,yP):P\in S_C\}.
\]

The conclusion is information-theoretic. A classical optimizer, a white-box
attacker and a coherent quantum algorithm receive identical observations for all
members of the orbit.

## 3. Exact orbit size

Two channel permutations can produce the same ordered object when some complete
channel records are identical. Define the signature of channel \(j\) as the
concatenation of its full input history and full target horizon. If the distinct
signatures have multiplicities

\[
m_1,\ldots,m_r,
\qquad
\sum_i m_i=C,
\]

then the number of distinct labeled objects in the orbit is

\[
|\mathcal F|=\frac{C!}{\prod_i m_i!}.
\]

Under a uniform prior over this orbit, exact ordered recovery satisfies

\[
P^*_{\rm exact}\le\frac{1}{|\mathcal F|}.
\]

For seven pairwise distinct ETTm1 variables this ceiling is

\[
\frac{1}{7!}=\frac{1}{5040}\approx1.984\times10^{-4}.
\]

## 4. Applicability to PatchTST and iTransformer

The theorem applies to the repository implementations when:

- PatchTST uses one shared encoder and one shared head across channels;
- iTransformer has no channel-position embedding or channel-specific head;
- learned per-channel RevIN affine parameters are disabled;
- dropout is disabled during observation;
- input histories and targets are permuted together.

Adjacent transpositions generate the entire symmetric group. The executable
benchmark therefore verifies, for every adjacent channel swap:

1. output permutation equivariance;
2. invariance of the full parameter-gradient tuple;
3. the exact orbit size of the observed private sample.

Checking a generating set is an implementation audit. The theorem itself covers
all permutations once equivariance and loss invariance hold exactly.

## 5. What breaks the theorem

The stated result does not apply unchanged when:

- targets are public in a fixed channel order and only the input is private;
- channel permutations are declared an acceptable recovery equivalence;
- learned parameters attach identities to channel positions;
- individual PatchTST heads are used;
- per-channel metadata or embeddings identify variables;
- the release includes an auxiliary channel-identity observation.

These cases must be modeled as different observation channels rather than used
to weaken the stated fibre after seeing results.

## 6. Security interpretation

The theorem exposes a design tradeoff. Removing channel-identity parameters gives
useful permutation equivariance and allows models to process varying variable
orders, but it also prevents a full gradient from identifying the original
labeled ordering when both histories and targets are private.

This is not an optimization failure. Even a perfect attack can recover at most an
orbit representative unless additional channel-identity information is released.

## 7. Executable mapping

`qrecon.theory.channel_permutation` provides:

- `channel_permutation_orbit_size`;
- `channel_permutation_fibre_bound`;
- `ChannelPermutationFibreBound`.

`qrecon.benchmarks.channel_permutation_fibre` provides:

- exact tensor signatures for each channel history/target pair;
- adjacent-transposition output checks;
- full-gradient invariance checks;
- Wilson/bootstrap summaries;
- publication gates for pinned real datasets.

## 8. Claim boundary

The primary result is exact labeled-order non-identifiability when both the input
history and forecast target are private. If the evaluation target quotients
channel permutations, the relevant success metric is orbit-equivalence recovery,
not exact ordered recovery. If ordered targets are known, a separate known-target
analysis is required.
