# Theory-to-Experiment Evaluation Protocol

## 1. Objective

Experiments must test theorem assumptions, not merely produce attractive
reconstructions. Every result row should identify:

- candidate space and prior;
- target equivalence relation;
- access model;
- observation channel;
- number and size of observation fibres when enumerable;
- classical and quantum query budgets;
- oracle implementation cost and error;
- reconstruction success with uncertainty.

## 2. Finite-space validation suite

Start with enumerable candidate spaces before high-dimensional data.

### E1 — fibre bound

Enumerate all candidates, compute observations, construct fibres, and compare the
best empirical decoder success with

\[
\sum_y\max_{x\in F_y}\pi(x).
\]

The measured optimum must equal the executable bound up to numerical tolerance.

### E2 — noisy channel

Inject a known stochastic channel and verify

\[
P^*=\sum_y\max_x\pi(x)W(y\mid x).
\]

Then apply a stochastic post-processing kernel and verify that success does not
increase.

### E3 — local/global gap

Construct examples with:

- full local Jacobian rank and a distant collision;
- rank deficiency but an injective map;
- small singular value and noise-sensitive inversion.

These examples prevent the paper from treating numerical rank as a global
criterion.

### E4 — query curves

For candidate populations \(N\) and marked counts \(K\), report:

- exact classical success without replacement;
- ideal standard-Grover success by iteration;
- minimum queries for a common target success;
- cases where fixed-phase standard Grover cannot hit the requested probability
  at an integer iteration.

### E5 — cost break-even

Sweep compilation/setup cost, cost per oracle, shots, and shared workload \(M\).
Plot the exact boundary

\[
S_Q-S_C=M(V_C-V_Q).
\]

Do not label a region “advantage” unless the same cost unit and success target are
used on both sides.

## 3. Reconstruction benchmark matrix

After the finite suite passes, expand across:

- access: full per-sample gradient, partial layers, aggregate gradient, update,
  parameter release, classical API, coherent oracle;
- batch size: 1, 2, 4, 8, 16;
- models: logistic regression, MLP, LeNet, deeper CNN, then token/graph models;
- defenses: clipping, quantization, additive noise, aggregation, and privacy
  mechanisms;
- priors: direct, matched-size classical generator, VQC prior, structured finite
  candidate set;
- seeds: enough independent victim training and attack initializations for
  confidence intervals.

## 4. Required metrics

### Exactness and identity

- exact bit/word equality where quantized;
- `within_1e-6` for floating-point diagnostic results;
- original-sample top-1 success;
- success modulo the declared equivalence relation;
- fibre size and number of non-equivalent feasible candidates;
- false-positive rate among low-energy candidates.

### Approximate reconstruction

- MSE/relative L2;
- PSNR and SSIM for images;
- LPIPS only as a supporting perceptual measure;
- token edit distance or graph edit/isomorphism-aware measures as appropriate.

### Computation

- classical verifier calls;
- coherent oracle calls;
- wall-clock as a separate engineering metric;
- logical qubits, peak ancillas, T-count, T-depth, circuit depth;
- state preparation, shots, measurement, and decoding;
- setup/compilation amortization.

## 5. Statistical protocol

For stochastic experiments:

- predeclare primary metrics;
- report mean, median, standard deviation, and confidence intervals;
- include attack failures and timeouts rather than dropping them;
- pair methods on the same victim models and samples;
- use multiple attack restarts and retain both best-objective and final-iterate
  results;
- publish per-sample records in addition to aggregates.

## 6. Baselines

At minimum compare against:

- analytic leakage identities where applicable;
- DLG/iDLG-style gradient matching;
- stronger modern gradient inversion implementations;
- direct pixel/value optimization;
- matched-parameter classical generators;
- exhaustive or random search on finite candidate spaces;
- quantum-inspired search or tensor-network baselines when relevant.

A VQC-vs-small-MLP comparison alone is not evidence of quantum advantage.

## 7. Stop conditions for claims

Do not claim global identifiability when only a local rank certificate is known.

Do not claim original training-data recovery when several non-equivalent
candidates satisfy the verifier.

Do not claim coherent query advantage under a classical API.

Do not claim end-to-end advantage when oracle construction, precision, shots,
state preparation, or readout is omitted.

Do not claim robustness from a single seed, sample, architecture, or defense
setting.
