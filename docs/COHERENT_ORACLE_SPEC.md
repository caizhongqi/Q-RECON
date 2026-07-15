# Coherent Oracle Compiler Specification

## 1. Purpose

This specification defines the artifact required before Q-RECON may claim a
coherent-query result. A variational quantum prior is not a substitute for this
oracle. The compiler must translate a precisely quantized classical model into a
clean reversible implementation with testable semantics and an auditable
resource report.

## 2. Supported first milestone

The first complete compiler target should be intentionally narrow:

1. binary or multiclass logistic regression;
2. fully connected networks with affine layers and ReLU/comparison activations;
3. integer or fixed-point inputs and weights;
4. bounded candidate spaces small enough for exhaustive truth-table validation.

CNN, transformer, and floating-point support should follow only after this
milestone is correct.

## 3. Quantized model semantics

A model package must include:

- input bit width \(b_x\) and scale \(s_x\);
- weight bit width \(b_w\) and scale \(s_w\);
- bias bit width and scale;
- accumulator widths for every layer;
- signed representation, initially two's complement;
- rounding rule, initially round-to-nearest with deterministic tie handling;
- overflow rule;
- activation and output quantization.

For the first implementation, prefer **range-proved no-overflow arithmetic**.
Range analysis chooses accumulator widths large enough that all valid candidate
inputs remain in range. This avoids conflating modular wraparound with the
intended classical model.

The compiler must expose a pure reference evaluator

\[
f_b:\{0,1\}^{n_x}\to\{0,1\}^{n_y}
\]

whose bit-level output is the source of truth for circuit tests.

## 4. Clean value oracle

The required value oracle is

\[
U_{f_b}|x\rangle|y\rangle|0^a\rangle
=|x\rangle|y\oplus f_b(x)\rangle|0^a\rangle,
\]

where all \(a\) work qubits return to zero. A circuit that leaves
input-dependent garbage is not a clean oracle and cannot be inserted into
standard amplitude amplification without accounting for that garbage.

The implementation pattern is:

1. **compute** \(f_b(x)\) into work/output registers;
2. **copy or XOR** the final output into the designated output register;
3. **uncompute** all intermediate values by reversing the compute network.

## 5. Verification and phase oracles

A reconstruction score should be discretized as

\[
E_b(x)\in\{0,\ldots,2^{b_E}-1\}.
\]

Given a public threshold \(\tau\), define

\[
v_\tau(x)=\mathbf 1[E_b(x)\le \tau].
\]

The clean verifier is

\[
U_v|x\rangle|z\rangle|0^a\rangle
=|x\rangle|z\oplus v_\tau(x)\rangle|0^a\rangle.
\]

The corresponding phase oracle is obtained by phase kickback:

\[
O_v|x\rangle=(-1)^{v_\tau(x)}|x\rangle.
\]

If multiple candidates satisfy the threshold, the algorithm has recovered a
member of a feasible set, not necessarily the original training sample. The
paper must report \(K\), collision/equivalence structure, and original-sample
metrics separately.

## 6. Compiler intermediate representation

A minimal quantized IR should contain:

- `InputWord(shape, bits, scale, signed)`;
- `Affine(weight_words, bias_words, accumulator_bits)`;
- `ReLU(sign_bit_policy)`;
- `CompareLE(threshold_word)`;
- `Quantize(source_scale, target_scale, rounding)`;
- `OutputWord(bits, scale, signed)`.

Each IR operation must provide:

1. a classical reference implementation;
2. input/output range propagation;
3. a reversible lowering rule;
4. ancilla requirements;
5. symbolic and instantiated resource counts;
6. an inverse/uncompute construction.

## 7. Correctness obligations

### O1 — reference equivalence

For every valid candidate input \(x\), circuit measurement in the computational
basis must equal the reference evaluator bit for bit.

### O2 — reversibility

The compiled circuit must be a permutation of computational-basis states before
optional phase operations.

### O3 — clean ancillas

All work registers must end in their initial state for every valid input.

### O4 — phase correctness

For every basis candidate, the phase oracle must apply phase \(-1\) exactly when
the reference verifier returns one.

### O5 — precision contract

If approximate rotations or arithmetic are used, the compiler must return a
certified or upper-bounded operational error \(\delta\) compatible with the
hybrid bound in `THEORY_FOUNDATIONS.md`.

## 8. Required tests

For small input widths, tests must exhaustively enumerate every candidate and
check:

- reference output equals circuit output;
- all ancillas are zero after execution;
- applying the oracle and its inverse returns the initial state;
- the verifier bit matches the threshold predicate;
- phase signs match the verifier truth table;
- resource counts are deterministic and non-negative.

For larger widths, add randomized property tests but retain at least one fully
exhaustive configuration per supported operation.

A test that checks only a final reconstruction metric is insufficient to certify
an oracle.

## 9. Resource report

Every compiled oracle must emit a machine-readable record containing:

```text
input_qubits
output_qubits
peak_ancillas
logical_qubits
affine_operations
comparators
Toffoli_count
T_count
T_depth
Clifford_count
circuit_depth
inverse_calls_per_verifier
state_preparation_cost
precision_or_error_bound
```

Counts must state the gate set and decomposition convention. Parametric rotations
must include synthesis precision. If a simulator-specific macro is counted as
one gate, a second decomposed count is required.

## 10. Threat-model requirements

Q-Access must be justified independently of a normal model API. Acceptable paper
scenarios may include:

- a model owner intentionally exposes a coherent inference primitive;
- a quantized model is public and the attacker bears the compiler cost;
- a shared quantum service provides reversible access under a specified API;
- a theoretical oracle separation is studied explicitly as such.

The paper must not imply that sending classical prompts or tensors to a remote
service permits superposition queries.

## 11. Completion criterion

The compiler milestone is complete only when one end-to-end example satisfies
all of the following:

1. a quantized model and candidate verifier have fixed bit-level semantics;
2. the clean value and phase oracles pass exhaustive tests;
3. ideal Grover evolution returns the marked candidate distribution predicted by
   theory;
4. the full resource record is generated;
5. the classical exhaustive/random-search baseline uses the same verifier;
6. the cost comparison includes compilation, oracle calls, and readout;
7. no claim of original-sample recovery is made when the verifier has multiple
   non-equivalent marked candidates.
