# Independent UCI Appliances Channel-Privacy Evidence

## Scope

This experiment tests the focused anonymous-channel theorem on a real multivariate
source outside the ETT dataset family.

- Dataset: UCI Appliances Energy Prediction
- DOI: `10.24432/C5VC8G`
- License: CC BY 4.0
- Official archive SHA256:
  `2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a`
- Extracted CSV SHA256:
  `2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d`
- Selected semantic channels: `T1,T2,T3,T4,T5,T6,T7`

The source is downloaded from the official UCI archive and both the archive and
extracted CSV are checked before any experiment runs.

## Experimental contract

- context: 16
- forecast horizon: 4
- window stride: 8
- public labeled calibration windows: 0--19 in the primary point experiment
- private side-information evaluation windows: 20--39 in the primary experiment
- fibre/release windows: 40--59
- side-information permutations per evaluation window: 5
- release audit: float64 with tolerance `1e-10`
- release variants: full gradient, global clipping at 0.5, fixed signed 8-bit
  quantization at scale `1e-3`, Gaussian noise with standard deviation 0.01 under
  shared audit randomness, and first-parameter-only visibility

The two victim families are an iTransformer and a shared-head,
channel-independent PatchTST. Both use non-affine RevIN so that channel indices do
not enter through channel-specific learned parameters.

## Fibre and release results

Workflow run `29414389023` completed successfully. Artifact
`8342524508` has digest:

```text
sha256:7a15dcfac0c1023ef4bab8f64071d17b9037fe1d0720a11dc109986d633a3219
```

All publication gates passed.

| architecture | fibre points | orbit | ordered-success ceiling | maximum float32 gradient witness error | release checks |
|---|---:|---:|---:|---:|---:|
| iTransformer | 20 | 5040 | `1/5040` | `1.1920929e-6` | 20/20 |
| shared-head PatchTST | 20 | 5040 | `1/5040` | `7.1525574e-7` | 20/20 |

Every evaluated record has seven distinct channel signatures, hence the full
permutation orbit has size

\[
7! = 5040.
\]

Under a uniform prior over the residual orbit, exact labeled-order recovery from
the declared observation is bounded by

\[
1/5040 \approx 1.9841\times10^{-4}.
\]

The strict float64 release audit had maximum raw-gradient discrepancy
`8.8817842e-16` for both architectures. Every release variant was certified in
20/20 points for each architecture. Fixed 8-bit quantized releases were
bit-identical.

## Public calibration side information

The attacker is deliberately strengthened: it receives the exact numerical orbit
representative plus 20 disjoint, correctly labeled public calibration windows.
For 20 private windows and five hidden permutations per window, the declared
feature-assignment attacker obtained:

- exact labeled order: 0/100;
- 95% Wilson upper bound for exact-order success: 3.699%;
- mean correctly identified channel fraction: 19.286%;
- 95% bootstrap interval for mean channel accuracy: 16.429%--22.286%;
- median channel accuracy: 14.286%.

This empirical attacker is not Bayes-optimal. Its failure must not be interpreted
as a universal guarantee against arbitrary semantic side information. Full public
channel identities collapse the permutation ambiguity by definition.

## Calibration-size phase diagram

A second publication workflow fixes one disjoint private evaluation set and the
same five hidden permutation seeds while varying the number of public labeled
calibration windows. Workflow run `29414849906` and artifact `8342686065` have
artifact digest:

```text
sha256:f1cd549396597544dca25e6edd2f9f38a6f8fc9415791178cce9f94eb6f20f05
```

All 700 declared trials completed and all quality gates passed.

| public calibration windows | exact semantic order | mean channel accuracy | 95% bootstrap interval |
|---:|---:|---:|---:|
| 1 | 0/100 | 12.857% | [10.568%, 15.286%] |
| 2 | 0/100 | 18.571% | [15.857%, 21.000%] |
| 5 | 0/100 | 15.000% | [12.286%, 17.714%] |
| 10 | 0/100 | 27.143% | [23.711%, 30.571%] |
| 20 | 0/100 | 21.429% | [18.286%, 24.571%] |
| 40 | 0/100 | 28.571% | [25.000%, 32.286%] |
| 60 | 0/100 | 23.571% | [19.429%, 27.857%] |

At every calibration level, the 95% Wilson upper bound for exact-order recovery
is 3.699%. The empirical curve is not monotone: more public windows do not
necessarily improve this fixed diagonal-Gaussian feature matcher. This is useful
negative evidence against presenting one calibration count as a universal
side-information threshold, but it is not a proof that a stronger attacker could
not exploit larger or different public metadata.

## Combined evidence consequence

The previous ETT evidence contains 220 real-data fibre points. This independent
UCI experiment adds 40 more, bringing the current total to 260 real-data fibre
points across five source files, two modern forecasting families, short and
long-context settings, release postprocessing, optimizer-transcript tests and
side-information studies. The additional calibration sweep contributes 700
fixed-evaluation side-information trials.

The UCI result strengthens external validity; it does not change the theorem's
threat model. The central claim remains conditional on private semantic channel
order and a channel-permutation-equivariant victim/loss pair. It does not imply a
positive quantum speedup, a coherent compiler for the full Transformer stack, or
privacy when channel-specific parameters or fully public semantic labels break
the orbit.
