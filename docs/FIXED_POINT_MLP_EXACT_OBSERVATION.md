# Exact-Observation Oracle for Fixed-Point MLP Inversion

## 1. Scope

This artifact closes one previously missing semantic link in Q-RECON: the
classical inversion baseline and the coherent search oracle can now solve the
**same exact-output problem** for a two-layer fixed-point network.

The supported network is

\[
h(x)=\operatorname{ReLU}(Q_h(W_hx+b_h)),
\qquad
F(x)=Q_o(W_oh(x)+b_o),
\]

where all inputs, weights, biases, intermediate values, rounding rules, signed
representations, and overflow policies are defined by `FixedPointFormat` and
`QuantizedAffineLayer`. The current structure-preserving compiler requires the
range contracts used by the fixed-point value oracle; unsupported arithmetic is
rejected rather than silently approximated.

Given a public observed output vector

\[
t=(t_1,\ldots,t_m),
\]

Q-RECON constructs the predicate

\[
v_t(x)=\mathbf 1[F(x)=t].
\]

This is an observation-consistency predicate. It does not assert that every
consistent candidate equals the original private record.

## 2. Packed target semantics

Each output code is encoded with the declared final `FixedPointFormat`. If each
output uses `b` bits, the packed word is

\[
T=\sum_{j=0}^{m-1}\operatorname{word}(t_j)2^{jb}.
\]

The packing order is exactly the order used by
`ReversibleFixedPointMLPValueOracle.evaluate_input_word`. Target length and code
range are validated before any circuit is constructed.

## 3. Clean equality-oracle theorem

Let the existing clean value oracle satisfy

\[
U_F|x\rangle|y\rangle|0^a\rangle
 =|x\rangle|y\oplus F_{\mathrm{word}}(x)\rangle|0^a\rangle.
\]

Let `EQ_T` be the clean constant-equality network that toggles one target qubit
iff its source register equals `T`. The compiled circuit is

\[
U_{v_t}=U_F^{-1}\,EQ_T\,U_F.
\]

### Theorem 1 — exact predicate and clean work

For every valid input word `x`, target bit `z`, and zero-initialized work
register,

\[
U_{v_t}|x\rangle|z\rangle|0\rangle
 =|x\rangle|z\oplus\mathbf 1[F(x)=t]\rangle|0\rangle.
\]

#### Proof

The first application of `U_F` writes the bit-exact packed output into the value
register while returning the value oracle's internal work to zero. `EQ_T`
toggles the target exactly when that packed word equals `T` and returns its own
comparison work to zero. Applying `U_F^{-1}` clears the packed value register and
restores all value-oracle work. The input and target bit remain as stated.
Because every component is a reversible basis permutation, linear extension
establishes the same action on arbitrary superpositions. \(\square\)

The implementation is
`ReversibleFixedPointMLPEqualityOracle`; exhaustive basis tests independently
compare its circuit action with the public classical evaluator.

## 4. Phase-oracle corollary

Preparing the target qubit in

\[
| - \rangle=(|0\rangle-|1\rangle)/\sqrt 2
\]

turns the clean value predicate into

\[
O_t|x\rangle=(-1)^{v_t(x)}|x\rangle.
\]

The regression suite executes this phase semantics through the compiled circuit
and checks that Grover's measured success agrees with the exact analytic curve.
The marked-input list is used only to score success, not to implement phase
flips.

## 5. Resource composition

Let the complete clean MLP value circuit have gate counts
`X_F`, `CNOT_F`, `Toffoli_F`, depth `D_F`, and peak internal work `A_F`. Because
the equality oracle applies the value circuit and its inverse, the value portion
contributes twice these gate counts and depth.

For a packed output width `B`, the current equality ladder contributes:

- `2 z(T)` X gates, where `z(T)` is the number of zero bits in the target word;
- one CNOT when `B=1`;
- one Toffoli when `B=2`;
- `2B-3` Toffoli gates and `B-2` reusable clean ancillas when `B>=3`.

Under the repository's declared exact Toffoli accounting convention,

\[
T\text{-count}\le 7N_{\mathrm{Toffoli}},
\qquad
T\text{-depth}\le 3N_{\mathrm{Toffoli}}.
\]

These are auditable upper bounds, not claims of optimal synthesis. Any optimized
backend must preserve the same bit semantics and report its own decomposition
contract.

## 6. Matched classical boundary

`solve_fixed_point_mlp_exact_output` is a complete branch-and-bound solver when
`max_solutions=None`. It propagates sound output intervals through the same
fixed-point layers and prunes a partial assignment only when the target lies
outside the enclosure. At leaves it evaluates the same
`QuantizedAffineLayer.evaluate_codes` semantics used by the compiler.

For a common candidate domain, the required cross-check is

\[
\{x:\text{branch-and-bound returns }x\}
 =\{x:U_{v_t}\text{ marks }x\}.
\]

The automated test suite verifies this equality on a multi-output MLP and also
checks the exhaustive basis permutation, inverse action, ancilla cleanup,
phase signs, Grover success, and resource consistency.

## 7. Candidate-domain warning

The circuit's default search register spans every bit word admitted by the input
format. The classical solver can additionally receive arbitrary per-feature
subdomains. A matched comparison therefore requires one of the following:

1. use the complete fixed-point word domain on both sides;
2. include a clean domain-membership predicate in the quantum verifier; or
3. account for a state-preparation method that creates a uniform superposition
   over exactly the declared structured domain.

Ignoring this distinction would give the classical and quantum methods different
candidate priors and invalidate a cost comparison.

## 8. Identifiability and collision reporting

Exact output equality is generally many-to-one. Before interpreting a marked
candidate as a reconstruction, an experiment must report:

- candidate population and prior;
- marked count `K`;
- exact output-fibre size or a certified estimate;
- original-record success;
- success modulo a predeclared target equivalence;
- Bayes exact/equivalence-class ceilings.

Grover amplification finds a marked member. It does not distinguish candidates
inside one non-identifiable fibre.

## 9. Claim boundary

This milestone establishes a clean, non-enumerative exact-observation verifier
for the supported fixed-point MLP and connects it to a complete classical solver
on the same semantics. It still does **not** establish end-to-end quantum
advantage. Such a claim additionally requires:

- a nontrivial task without a cheaper analytic decoder;
- the strongest SAT/SMT, MIP, branch-and-bound, meet-in-the-middle, and optimized
  inversion baselines applicable to that task;
- matched candidate domains and preprocessing rights;
- state preparation, inverse calls, diffusion, fault-tolerant gates,
  measurement, and decoding costs;
- unknown-`K` or certified-`K` search protocol;
- a robust nonempty break-even region under predeclared cost uncertainty.

A rigorous no-advantage boundary remains a valid top-tier result if the matched
classical solver or oracle construction cost eliminates the apparent query
speedup.
