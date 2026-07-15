# Exact SMT Inversion for Fixed-Point MLPs

## 1. Role in Q-RECON

A quantum reconstruction claim is not credible if the coherent verifier is
compared only with random search or exhaustive enumeration. The same discrete
network often exposes arithmetic structure that a SAT/SMT or mixed-integer
solver can exploit.

`qrecon.oracles.z3_inversion` provides an exact Z3 baseline for the same
two-layer fixed-point model used by the clean MLP value/equality oracle:

\[
h(x)=\operatorname{ReLU}(Q_h(W_hx+b_h)),
\qquad
F(x)=Q_o(W_oh(x)+b_o).
\]

For a declared finite product domain and observed output `t`, the solver
enumerates

\[
\{x:F(x)=t\}.
\]

The optional dependency is installed with

```bash
pip install -e '.[solver]'
```

## 2. Exact symbolic semantics

The SMT encoding uses Z3 integer variables for input codes. Each variable is
constrained to its caller-declared finite code domain. Every affine accumulator
is represented as an unbounded integer expression, so no host-language or solver
bit-vector wraparound is introduced implicitly.

### 2.1 Bias alignment

If the affine product has `f_x+f_w` fractional bits and the bias has `f_b`, the
bias is aligned using the same `rescale_code` routine as the classical reference
evaluator. Because weights and biases are public constants, this transformation
is performed exactly before the symbolic expression is built.

### 2.2 Requantization

For a right shift by `s>0`, Q-RECON uses nearest rounding with ties away from
zero:

\[
R_s(a)=\operatorname{sgn}(a)
\left\lfloor\frac{|a|+2^{s-1}}{2^s}\right\rfloor.
\]

The SMT expression uses an `If` for the sign and integer division of the
nonnegative magnitude, matching `round_shift_right` exactly. Upscaling is exact
multiplication by a power of two.

### 2.3 Activation and overflow

ReLU is encoded as

\[
\max(0,z)=\operatorname{If}(z<0,0,z).
\]

For an output format with `overflow="raise"`, range constraints require

\[
q_{\min}\le z\le q_{\max}.
\]

For `overflow="saturate"`, the expression is clamped with nested `If` terms.
No candidate is silently interpreted using a different modular arithmetic.

## 3. Correctness theorem

### Theorem 1 — reference/SMT equivalence

Fix valid layer definitions, finite input domains, and a target output vector
`t`. For every candidate `x` in the declared domain, the SMT formula is satisfied
by `x` if and only if

\[
\texttt{output\_layer.evaluate\_codes(}
\texttt{hidden\_layer.evaluate\_codes(x))}=t.
\]

#### Proof

Input-domain clauses are exact finite membership tests. Each affine expression is
the same integer weighted sum and aligned bias used by the reference evaluator.
Section 2.2 gives an algebraically identical encoding of deterministic rescaling;
Section 2.3 gives identical ReLU and overflow behavior. Composition therefore
produces the same code at every hidden and final output. Equality clauses hold
exactly when the reference output equals `t`. \(\square\)

The CI suite verifies this theorem extensionally for every reachable output of a
fractional fixed-point MLP over a 64-candidate domain. It separately checks
round-half-away-from-zero, ReLU, and saturating overflow.

## 4. Complete solution enumeration

After each satisfying model, the solver adds a blocking clause

\[
\bigvee_i x_i\ne x_i^*,
\]

then checks again. If the final result is `unsat` and no caller solution limit was
used, the returned set is complete. Reports distinguish three terminations:

- `exhausted`: all satisfying candidates were enumerated;
- `solution_limit`: a caller limit stopped enumeration, so completeness is false;
- `unknown`: timeout or another solver limitation prevented a proof.

A partial result is never labeled complete. Reports include solver checks,
encoded constraint count, candidate count, solution count, timeout, and the
solver's unknown reason when applicable.

## 5. Three-way solution-set audit

For tractable instances Q-RECON requires the following equality on one declared
candidate domain:

\[
S_{\mathrm{SMT}}=S_{\mathrm{BnB}}=S_{\mathrm{oracle}},
\]

where

- `S_SMT` is returned by `solve_fixed_point_mlp_with_z3`;
- `S_BnB` is returned by the sound interval branch-and-bound solver;
- `S_oracle` is the set marked by the clean fixed-point MLP equality oracle.

The coherent register naturally spans the complete fixed-point word space. If
the declared classical domain is smaller, the oracle set must be filtered only
for auditing; a real quantum comparison must instead include structured state
preparation or a clean domain-membership predicate. This distinction is emitted
by `examples/z3_fixed_point_inversion.py` rather than hidden.

## 6. Classical baseline protocol

A paper experiment should run at least:

1. branch-and-bound with sound interval propagation;
2. Z3 SMT with complete enumeration or a declared timeout;
3. optimized exhaustive evaluation for small reference instances;
4. task-specific algebraic, meet-in-the-middle, MIP, or gradient baselines when
   their structure applies.

For each method report:

- reusable preprocessing;
- online time and peak memory;
- complete/incomplete status;
- number of solver checks or explored nodes;
- solution set and fibre size;
- failure and timeout rates across seeds/instances;
- scaling with input width, hidden width, precision, and domain size.

SMT is a stronger opponent than random search, but it is not automatically the
strongest possible opponent for every network.

## 7. Claim boundary

This module strengthens the classical boundary; it does not establish quantum
advantage. A valid advantage region must remain nonempty after substituting the
best observed classical solver, including its preprocessing and amortization,
and after quantum compilation, state preparation, unknown-`K` search,
verification, measurement, error correction, and readout are priced in the same
unit.

A result showing that SMT or another structured solver removes the apparent
Grover advantage is a rigorous no-advantage result and can be scientifically more
important than an artificial positive comparison.
