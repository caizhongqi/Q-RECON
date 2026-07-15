# Q-RECON CCF-A Evidence Ledger

## 1. Purpose

This ledger records exactly which Q-RECON claims are supported by immutable theorem
statements, source code, tests, real-data experiments, and archived GitHub Actions
artifacts. It is an internal submission gate, not a prediction or guarantee of CCF-A
acceptance.

A result is counted as **validated** only when all of the following hold:

1. the threat model and recovery target are explicit;
2. the theorem or algorithm is present in the repository;
3. failures remain in the denominator and incomplete solvers are identified;
4. the dataset revision or file SHA256 is locked;
5. the relevant quality gate passes;
6. the workflow and artifact identifiers are recorded below.

## 2. Current primary thesis

The strongest coherent paper thesis is now a negative information-theoretic result:

> Anonymous-channel equivariant modern forecasting Transformers have exact
> channel-permutation training-data fibres. Exact labeled channel order is not
> identifiable from their released training gradients. This ambiguity is shared by
> classical and coherent quantum algorithms, survives deterministic multi-step
> first-order training, and is closed under clipping, quantization, partial gradient
> visibility, and data-independent randomized release mechanisms.

This is distinct from two weaker statements:

- a gradient-matching optimizer sometimes fails to converge;
- a quantum search implementation has no practical speedup.

The primary result is an exact observation-equivalence theorem. The Bayes ceiling is
caused by absent information, not by computational hardness.

## 3. Formal theorem chain

### T1 — full-gradient invariance

For a channel-permutation equivariant forecaster

\[
f_\theta(XP)=f_\theta(X)P
\]

trained by channel-symmetric mean squared error,

\[
\nabla_\theta\mathcal L(\theta;XP,YP)
=
\nabla_\theta\mathcal L(\theta;X,Y)
\]

for every parameter value, batch size, and channel permutation.

### T2 — exact orbit and Bayes ceiling

If complete private channel signatures have multiplicities
\(m_1,\ldots,m_r\), the number of distinct labeled private objects in the orbit is

\[
|\mathcal O|=\frac{C!}{\prod_jm_j!}.
\]

Under a uniform prior, exact labeled-order recovery is bounded by

\[
P^*_{\mathrm{ordered}}=\frac{1}{|\mathcal O|}.
\]

For seven distinct ETT channels this is `1/5040`.

### T3 — classical and coherent quantum indistinguishability

Every orbit member induces the same gradient function for every parameter query and
therefore the same clean coherent unitary constructed from that function. Adaptive
classical transcripts and quantum final states are identical across the orbit. No
number of queries can exceed the Bayes ceiling without an additional side-information
channel.

### T4 — deterministic training-transcript closure

From common initial parameters and optimizer state, deterministic SGD, momentum,
Adam, and AdamW produce identical gradients, optimizer states, checkpoints, and final
model deltas on `(X,Y)` and `(XP,YP)`.

### T5 — release-channel closure

Every deterministic function of an invariant gradient remains invariant. Every
randomized release kernel whose law depends on the private object only through that
gradient has the same conditional output distribution on the entire orbit. The
implemented audits cover:

- full exact gradients;
- global norm clipping;
- fixed symmetric quantization;
- partial parameter visibility;
- independent additive Gaussian noise;
- combinations of those mechanisms.

The proofs, architecture scope, executable mappings, and claim boundaries are in
`CHANNEL_PERMUTATION_NONIDENTIFIABILITY.md`.

## 4. Validated modern-model real-data evidence

### 4.1 ETTm1: theorem, control, optimizer, and release study

- Workflow run: `29409084808`
- Artifact: `ettm1-channel-permutation-publication`
- Artifact id: `8340339796`
- Artifact digest:
  `sha256:38e1c26766b23f98179a85e2ed2da9e71f277c286e8bd90b49e659bbecdf20c9`
- Immutable ETTm1 file SHA256:
  `6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e`

Validated results:

- anonymous-channel iTransformer: generator-complete full-gradient fibre on 20/20
  windows;
- shared-head channel-independent PatchTST: full-gradient symmetry on 20/20 windows;
- channel-specific PatchTST head: symmetry broken on 20/20 windows;
- three-step AdamW gradient, optimizer-state, checkpoint, and model-delta equality on
  the two anonymous architectures;
- full, clipped, quantized, Gaussian-noisy, partial, and combined releases preserve
  the fibre on 20/20 windows;
- the seven-channel exact labeled-order ceiling is `1/5040` on the generic windows.

The float64 optimizer audit reports maximum full-gradient discrepancies below
`3e-15`; corresponding parameter, optimizer-state, and final-delta discrepancies are
zero at the reported precision. The channel-specific control produces nonzero
transcript differences, demonstrating that the certificate is sensitive to explicit
symmetry breaking.

### 4.2 ETTm2 and ETTh1: cross-dataset, cross-architecture replication

- Workflow run: `29409660763`
- Artifact: `ett-cross-dataset-channel-permutation`
- Artifact id: `8340583675`
- Artifact digest:
  `sha256:c9b1898ba62f6df3f49660468ed39783bb2129be185d947014a1780c33153e01`

The matrix contains 20 immutable windows in each of four cells:

1. ETTm2 × iTransformer;
2. ETTm2 × shared-head PatchTST;
3. ETTh1 × iTransformer;
4. ETTh1 × shared-head PatchTST.

All fibre and release quality gates pass. Every declared clipping, fixed 8-bit
quantization, Gaussian-noise, and partial-visibility check preserves the observation
fibre.

Orbit statistics:

- ETTh1: orbit size `5040` on 20/20 windows for both architectures;
- ETTm2: orbit size `5040` on 18/20 windows and `2520` on 2/20 windows because one
  pair of complete private channel signatures is duplicated; mean orbit size `4788`.

Combining ETTm1, ETTm2, and ETTh1 gives 120 primary real-data modern-model fibre
observations across two architecture families, excluding the additional
symmetry-breaking controls and optimizer/release repetitions.

## 5. Validated PatchTST reconstruction evidence

These experiments are classical white-box gradient inversion. They do not imply a
coherent PatchTST oracle or quantum advantage.

### 5.1 TS-Inverse-style objective baseline

- Workflow run: `29406724089`
- Artifact: `gifteval-patchtst-ts-inverse-publication`
- Artifact id: `8339705372`
- Artifact digest:
  `sha256:f97e2b0efc9d7ed9f789753098c5c5351249447d525349174cd2c3474e1cf7ec`
- Dataset: revision-pinned GIFT-Eval
- Victim: PatchTST
- Evaluation: 20 records, three restarts per record, no failed attempts

Selected-by-released-objective results:

- all-values-within-`0.1`: `9/20 = 45%`;
- relative-L2 below `0.5`: `18/20 = 90%`;
- mean MSE: `0.0864025`;
- median MSE: `0.0153445`;
- mean relative-L2: `0.199546`;
- mean correlation: `0.959453`;
- mean sMAPE: `31.05%`;
- bitwise exact recovery: `0/20`.

This establishes nontrivial modern-model leakage, but not exact reconstruction of all
records.

### 5.2 Paired learned-initializer confirmation

- Workflow run: `29409084824`
- Artifact: `gifteval-patchtst-paired-learned-publication`
- Artifact id: `8340366817`
- Artifact digest:
  `sha256:f4efbea69b7fd01e214487bf0d9e669c03a291107c04072942cf9a0f53456086`
- Confirmatory records: 20 records disjoint from the auxiliary calibration split
- Restarts: three per method and record

Random L1 initialization:

- tolerance success: `6/20 = 30%`;
- relative-L2 success: `19/20 = 95%`;
- mean MSE: `0.0927647`;
- mean relative-L2: `0.211065`.

Learned quantile initialization:

- tolerance success: `9/20 = 45%`;
- relative-L2 success: `13/20 = 65%`;
- mean MSE: `0.250209`;
- mean relative-L2: `0.323295`.

The learned initializer lowers MSE on 13/20 paired records but has catastrophic
failures. The paired mean learned-minus-random MSE is `+0.157445`, with 95% bootstrap
interval `[0.05608, 0.26943]`; the learned method is significantly worse in mean.
This negative result is retained and rules out a cherry-picked claim that the learned
initializer is uniformly stronger.

### 5.3 Matched attack and defense matrix

- Workflow run: `29409660770`
- Artifact: `gifteval-patchtst-attack-defense-confirmatory`
- Artifact id: `8340782512`
- Artifact digest:
  `sha256:2009123598dd5434f1730788e13ea6b33eb2296d324a7ccdeb6bf27ddbd6549b`
- Dataset: revision-pinned GIFT-Eval
- Victim: PatchTST, 366 trainable parameters
- Confirmatory records: indices 40–59
- Restarts: three per method/condition
- Attack attempts: 240, all completed
- Defense attempts: 300, all completed
- Attack, defense, dataset, and victim identity checks: all passed

Attack summary:

| attack | tolerance success | relative-L2 success | mean MSE | mean correlation |
|---|---:|---:|---:|---:|
| DLG L2 | 1/20 | 17/20 | 0.156227 | 0.921605 |
| InvG cosine | 0/20 | 18/20 | 0.186816 | 0.901958 |
| Q-RECON hybrid | 0/20 | 18/20 | **0.119616** | 0.940666 |
| temporal-prior hybrid | 0/20 | 18/20 | 0.126109 | 0.936189 |

No method dominates every metric. DLG obtains the only strict tolerance success;
Q-RECON hybrid has the lowest mean MSE among the four declared objectives.

Defense summary using the completed Q-RECON hybrid attack:

| release | relative-L2 success | mean MSE | mean correlation |
|---|---:|---:|---:|
| full exact gradient | 18/20 | 0.114705 | 0.943110 |
| Gaussian noise, std `1e-3` | 18/20 | 0.119428 | 0.938472 |
| global clipping, norm `1` | **12/20** | **0.242842** | 0.874246 |
| last head only | 17/20 | 0.145164 | 0.927122 |
| symmetric int8 | 18/20 | 0.103469 | 0.949040 |

Small independent Gaussian noise and the declared int8 release do not materially
reduce reconstruction in this configuration. Global clipping is the strongest
implemented defense in this matrix: it reduces relative-L2 success from 90% to 60%
and roughly doubles mean MSE. These are empirical results for the declared scale, not
universal defense guarantees.

## 6. Full current-head software validation

- Workflow run: `29410241236`
- Branch head validated: `9711de24e0647e5f1c3d31dbcdfd16c4562052fe`

All jobs completed successfully:

- complete test suite on Python 3.10;
- complete test suite on Python 3.12;
- fast executable contract checks;
- Python source compilation;
- focused exact Z3/branch-and-bound/coherent-verifier checks;
- differentiable PennyLane quantum-prior forward/backward smoke test.

## 7. Failure ledger

Failures are evidence and remain recorded.

1. The first learned-calibration workflow failed during test collection because the
   paired learned benchmark was not exported from `qrecon.benchmarks`. The public API
   export was fixed and the paired publication study subsequently passed.
2. A float32 multi-step AdamW audit amplified near-zero reduction residuals to a
   visible parameter difference. The formal equality theorem is exact; the
   certificate was changed to float64 rather than relaxing the tolerance. The
   ETTm1 optimizer study then passed with gradient discrepancies below `3e-15`.
3. The first defense matrix used unsupported match mode `l1` in the release-aware
   attack. All 300 attempts failed with the same retained error. The declaration was
   corrected to the supported hybrid objective and the complete 300-attempt rerun
   passed.
4. A redundant cross-dataset workflow used inconsistent report-hash encodings and a
   numerically brittle combined noise-then-quantization audit. It was removed rather
   than counted. The independent `ett-cross-dataset-publication` workflow passed the
   intended ETTm2/ETTh1 matrix.

## 8. Claims currently supported

### Supported

- exact full-gradient channel-permutation non-identifiability for anonymous-channel
  iTransformer and shared-head channel-independent PatchTST;
- exact orbit/Bayes ceilings for the declared labeled-order target;
- classical and coherent quantum indistinguishability from the same observation;
- deterministic SGD/momentum/Adam/AdamW transcript closure;
- closure under declared clipping, quantization, Gaussian noise, partial visibility,
  and repeated release channels;
- cross-dataset replication on ETTm1, ETTm2, and ETTh1;
- explicit symmetry-breaking controls;
- nontrivial but imperfect GIFT-Eval PatchTST reconstruction;
- a matched attack and defense matrix with all failures retained.

### Not supported

- practical or asymptotic end-to-end quantum advantage for PatchTST/iTransformer;
- a coherent reversible compiler for full attention, Softmax, LayerNorm, or RevIN;
- exact recovery of all PatchTST training records;
- non-identifiability when semantic channel labels or ordered targets are public;
- protection of the unordered orbit representative;
- universal effectiveness of clipping, noise, quantization, or partial gradients;
- hardware-comparable runtime speedups from GitHub-hosted runner timings.

## 9. Strict remaining submission gates

The negative-theorem route is technically much closer to a top-tier submission than
the positive-quantum-advantage route, but the repository must not be labeled
“CCF-A achieved.” Remaining gates are:

1. write a focused paper around one primary threat model rather than presenting the
   entire repository as one contribution;
2. compare the theorem explicitly with prior Transformer/time-series gradient
   inversion and symmetry-based privacy results;
3. add a side-information study distinguishing private ordered channel identities,
   public semantic labels, and recovery modulo permutation;
4. archive final publication artifacts under a stable release and freeze the Python
   environment;
5. obtain independent theorem/code review and resolve all review findings;
6. report GitHub-hosted timings only as execution records, not as hardware speedup
   evidence;
7. keep the positive quantum-advantage claim out of the paper unless a common-unit
   `C_Q < C_C` region is later demonstrated against the strongest matched classical
   pipeline.

A rigorous negative paper can be top-tier without a practical quantum speedup. The
current defensible contribution is that a broad class of modern forecasting models
loses exact labeled-channel information before either a classical or quantum attack
begins.
