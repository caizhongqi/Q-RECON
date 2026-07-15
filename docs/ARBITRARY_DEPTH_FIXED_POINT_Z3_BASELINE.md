# Arbitrary-Depth Fixed-Point SMT Inversion Baseline

## 1. Purpose

A structure-preserving coherent verifier must be compared with a classical solver
that receives the same model, candidate domain, target observation, arithmetic
semantics, and stopping contract. Restricting the classical opponent to a
two-layer network after extending the coherent compiler to arbitrary depth would
create an artificial comparison.

`qrecon.oracles.z3_deep_inversion.solve_fixed_point_deep_mlp_with_z3` therefore
encodes the same arbitrary-depth `QuantizedAffineLayer` sequence as the coherent
value/equality oracle.

## 2. Matched semantics

For every layer, the SMT encoding uses:

- unbounded integer accumulation;
- declared fixed-point input, weight, bias, and output scales;
- deterministic nearest rounding with half ties away from zero;
- the declared `identity` or `ReLU` activation;
- range constraints for `overflow="raise"`;
- endpoint clamping for `overflow="saturate"`;
- exact finite input code domains;
- raw final-output code equality.

Adjacent dimensions and word formats are validated by `QuantizedNetwork`. The
final layer must use identity activation, matching the arbitrary-depth coherent
value and exact-output oracle contract.

## 3. Exact fibre enumeration theorem

Let `D=D_1 x ... x D_d` be the declared finite input-code domain, let

\[
f:D\rightarrow\mathcal Y
\]

be the public bit-exact quantized network evaluator, and let `t` be a target code
vector. The SMT formula contains:

1. one integer variable `x_i` constrained to membership in `D_i`;
2. one symbolic expression for every neuron under the matched layer semantics;
3. final constraints `f_j(x)=t_j` for every output coordinate.

After each satisfying model `x`, the solver adds the blocking clause

\[
\bigvee_i x_i\ne x_i^{(model)}.
\]

### Theorem — complete exact-output fibre

If Z3 eventually returns `unsat` and no solution limit is active, the reported
solution set equals

\[
F_t=\{x\in D:f(x)=t\}.
\]

#### Proof

Every reported model satisfies the domain and matched network constraints, hence
belongs to `F_t` (soundness). Conversely, each member of `F_t` satisfies every
encoded constraint. Blocking removes only models already returned, so any
unreported member remains satisfying until selected. A final `unsat` result
therefore proves that no member remains (completeness). `square`

The report uses termination `exhausted` only for this final-unsatisfiable case.
`solution_limit` and `unknown` are never presented as complete fibre
certificates.

## 4. Three-way independent cross-check

For small full-word domains, the test artifact compares:

1. direct enumeration through `QuantizedNetwork.evaluate_codes`;
2. the complete Z3 solution set;
3. marked basis words from
   `ReversibleFixedPointDeepMLPEqualityOracle`.

For every reachable output of a three-layer fixed-point ReLU network, all three
sets must agree exactly. The same test also verifies the clean reversible basis
permutation. This catches mismatches in rounding, signed representation,
activation, overflow, packing, target equality, or solver blocking.

The executable example emits the target, reference fibre, Z3 report, coherent
marked words, permutation result, and oracle resources as one JSON artifact.

## 5. Complexity and claim boundary

SMT model enumeration can be exponential in the worst case, but it is
structure-aware and can exploit arithmetic constraints without materializing a
complete truth table. Its preprocessing, solver checks, timeout, termination,
and solution count must be reported.

This baseline closes the depth mismatch between the coherent compiler and the
exact classical solver. It does not prove that Z3 is the strongest possible
classical method. The final task must also consider branch-and-bound, MIP,
algebraic elimination, meet-in-the-middle, dynamic programming, and optimized
continuous inversion when their assumptions apply. Quantum advantage may be
claimed only against the best completed matched baseline, not merely against
this SMT formulation.
