# Q-RECON Theory Claim Matrix

The complete access-model, information, query, and cost arguments live in
[`THEORY_FOUNDATIONS.md`](THEORY_FOUNDATIONS.md), the aggregate-gradient collision
proof in
[`BATCH_GRADIENT_NONIDENTIFIABILITY.md`](BATCH_GRADIENT_NONIDENTIFIABILITY.md),
the exact single-record gradient theorem and compiler in
[`GRADIENT_RECONSTRUCTION_ORACLE.md`](GRADIENT_RECONSTRUCTION_ORACLE.md), the
coherent compiler contract in
[`COHERENT_ORACLE_SPEC.md`](COHERENT_ORACLE_SPEC.md), the exact finite compiler
proofs in
[`TRUTH_TABLE_ORACLE_BASELINE.md`](TRUTH_TABLE_ORACLE_BASELINE.md) and
[`ANF_ORACLE_OPTIMIZATION.md`](ANF_ORACLE_OPTIMIZATION.md), the polynomial
structure-preserving compiler proofs in
[`STRUCTURE_PRESERVING_AFFINE_ORACLE.md`](STRUCTURE_PRESERVING_AFFINE_ORACLE.md),
[`REVERSIBLE_MLP_ORACLE.md`](REVERSIBLE_MLP_ORACLE.md), and
[`DEEP_REVERSIBLE_MLP_ORACLE.md`](DEEP_REVERSIBLE_MLP_ORACLE.md), the matched-cost
rules in [`END_TO_END_COST_PROTOCOL.md`](END_TO_END_COST_PROTOCOL.md), and the
empirical protocol in
[`THEORY_EVALUATION_PROTOCOL.md`](THEORY_EVALUATION_PROTOCOL.md).

| Claim | Mathematical status | Executable evidence | Publication status |
|---|---|---|---|
| deterministic observation fibres impose an exact Bayes ceiling | proved for finite candidate spaces | `bayes_reconstruction_success` | ready as a foundational lemma, not a novelty claim |
| recovery modulo a declared equivalence relation has a class-posterior Bayes optimum | proved for deterministic and noisy channels | `bayes_equivalence_reconstruction_success`, `channel_bayes_equivalence_reconstruction_success` | ready as a formal evaluation rule |
| stochastic post-processing cannot increase exact guessing probability | proved by data processing | `postprocess_channel` plus tests | ready as a foundational lemma |
| full-column gradient Jacobian rank certifies local injectivity | standard inverse-function argument | `gradient_jacobian_report.full_column_rank` | ready only as a local certificate |
| rank deficiency proves non-identifiability | false in general | counterexample documented | must not be claimed |
| finite truth-table enumeration gives global collision fibres | exact for the enumerated candidate space | `analyze_finite_oracle` | ready as a finite-space certificate; does not extrapolate beyond the enumerated domain |
| biased linear-regression aggregate gradients identify the complete batch for `B>=2` | false when inputs and regression targets are both private | explicit `A 1 = 1` batch-mixing collision family and tests | non-identifiability theorem ready under its stated scope |
| the same batch-mixing theorem automatically covers fixed labels, softmax, or arbitrary networks | not proved | no supporting construction | must not be claimed |
| a full exact single-record biased-linear squared-loss gradient is uniquely invertible when its bias gradient is nonzero | proved by `x_i=(g_w)_i/g_b` and `t=w^T x+b-g_b` | analytic decoder, complete finite fibre enumeration and exhaustive tests | ready as a Q-RECON-specific identifiability theorem |
| the zero-gradient single-record case identifies the original record | false whenever more than one representable `x` satisfies `t=w^T x+b` | explicit all-zero fibre enumeration | must not be claimed; the Bayes ceiling applies |
| Grover gives meaningful advantage for full exact single-record biased-linear gradients | false under the declared arithmetic model | nonzero case has an `O(d)` classical analytic decoder; zero case is non-identifiable | negative no-advantage corollary ready |
| identical released quantum states cannot be distinguished | standard Helstrom result | `binary_helstrom_success` | ready as an information bound |
| a classical prediction API supplies coherent queries | false by access-model definition | access models documented | must not be claimed |
| standard Grover search gives a black-box quadratic query reduction | standard result under clean Q-Access | exact success/query helpers and tests | usable only with explicit oracle assumptions |
| mixed-polarity minterm synthesis implements a clean finite value oracle | proved for the declared truth-table construction | `TruthTableOracle`, independently executed gate netlist, exhaustive basis-permutation and self-inverse tests | ready as a correctness theorem for the exponential baseline |
| ANF synthesis implements the same clean finite value oracle | proved by uniqueness of the GF(2) multilinear polynomial | `ANFOracle`, Möbius transform, independent netlist execution and exhaustive cross-backend tests | ready as an exact optimization theorem |
| ANF always beats minterm synthesis | false | `compare_exact_syntheses` retains both records and chooses by declared resource key | must not be claimed |
| affine Boolean predicates can have zero Toffoli cost under ANF | proved from degree-at-most-one monomials | parity regression and scaling reports | ready under the stated decomposition convention |
| the finite verifier gives the declared phase sign through kickback | follows from clean one-bit XOR semantics | phase tests and Grover simulation using either exact backend | ready for the finite baseline |
| the truth-table-derived compiler family is asymptotically efficient | false in the worst case | resource and scaling reports expose exponential term counts | must not be claimed |
| an integer affine model can be compiled to a clean polynomial-size value or threshold oracle | proved for the declared two's-complement, no-overflow, shift-add and ripple-carry semantics | `ReversibleIntegerAffineValueOracle`, `ReversibleIntegerAffinePredicateOracle`, exhaustive adder/oracle tests and exact resource formulas | ready as the first structure-preserving compiler theorem |
| a two-layer integer `Affine-ReLU-Affine/Threshold` network can be compiled to a clean predicate and phase oracle | proved by clean suboracle composition, exact sign-controlled ReLU, and Bennett uncomputation | `ReversibleIntegerMLPPredicateOracle`, exhaustive candidate/target/inverse tests, Grover phase regression and exact composed gate counts | ready as a non-linear compiler theorem under the stated integer/two-layer scope |
| an arbitrary-depth integer ReLU MLP can be compiled with one shared arithmetic work region | proved by layerwise induction, reverse liveness cleanup, and maximum-work reuse | `ReversibleIntegerDeepMLPPredicateOracle`, exhaustive three-layer tests, exact maximum-versus-sum ancilla and gate-count identities | ready as the depth-generalization and qubit-liveness theorem under integer semantics |
| an exact single-record linear-training gradient can be compiled as a polynomial-size clean value/equality/phase oracle | proved by affine residual computation, signed modular variable multiplication, full-gradient copy/equality, and reverse cleanup | `ReversibleSingleRecordGradientValueOracle`, `ReversibleSingleRecordGradientEqualityOracle`, exhaustive multiplier/value/verifier/Grover tests | ready as the first structure-preserving training-leakage compiler theorem |
| the current structure compiler supports arbitrary fixed-point requantization | false | adapters explicitly reject fractional scaling | must not be claimed until a reversible rounding/shift compiler and error theorem exist |
| the present VQC prior gives quantum query advantage | not established | no supporting oracle experiment | must not be claimed |
| a structure-preserving compiled neural verifier preserves query advantage end to end | open | clean integer Affine, equality, arbitrary-depth ReLU MLP, and exact-gradient compilers plus a cost planner now exist; the first real leakage benchmark is classically invertible or non-identifiable | requires a harder identifiable leakage, strongest classical baseline, precision, state preparation and measured break-even evidence |
| batch-one biased first-layer Linear gradients reveal the raw input | proved under explicit leakage assumptions | analytic implementation and real-data verification | usable only with the stated assumptions |

## Target-equivalence Bayes theorem

Let `c(x)` denote the declared acceptable target class of candidate `x`. For a
finite prior `pi(x)` and noisy observation channel `W(y|x)`, the best possible
success probability for returning any candidate in the correct target class is

\[
P^*_{\mathrm{class}}(X\mid Y)
=\sum_y\max_a\sum_{x:c(x)=a}\pi(x)W(y\mid x).
\]

For each observed `y`, any decision rule must choose one class `a`. Its joint
correct mass is the inner class sum, so choosing the largest class posterior is
optimal independently for every `y`; summing over observations proves the
formula. The deterministic result follows by setting
`W(y|x)=1[g(x)=y]`.

This theorem is the correct formal target for datasets with permutation,
isomorphism, tokenization, or application-defined equivalences. Exact-candidate
success and class success must be reported separately.

## Exact finite compiler theorems

Let `f:{0,1}^n -> {0,1}^m` be the bit-level reference evaluator produced by the
quantized model semantics.

The mixed-polarity backend emits one fully controlled X for every set output bit
of every input minterm. Exactly one minterm matches a basis input. If
`S=sum_x wt(f(x))`, this backend emits exactly `S` terms and, for `n>=2`, uses
`S(2n-3)` Toffolis under the declared clean-ancilla decomposition.

The ANF backend computes the unique multilinear GF(2) polynomial of every output
bit by Möbius transform and emits one positive-control gate per nonzero monomial.
A degree-`d>=2` monomial uses `2d-3` Toffolis under the same convention; degree
zero and one use only X and CNOT. Both independently executed gate lists satisfy

\[
U_f|x\rangle|y\rangle|0^a\rangle
=|x\rangle|y\oplus f(x)\rangle|0^a\rangle.
\]

The two finite backends are exhaustive cross-checks and resource upper bounds.
Neither provides a polynomial worst-case neural-network compiler.

## Structure-preserving compiler theorems

The integer Affine backend computes constant products by sign/zero extension and
shift-add, accumulates them with clean ripple-carry addition, copies the result,
and reverses every arithmetic operation. Under its explicit no-overflow proof,
the modular bit circuit equals the mathematical integer affine map on the full
input-word domain.

For a two-layer MLP, the compiler composes a clean first Affine value oracle, a
reversible componentwise ReLU copy, and a clean final Affine-threshold oracle. It
then reverses ReLU and the first Affine oracle. For hidden width `w>1` and `h`
hidden neurons, ReLU compute/uncompute contributes exactly `4h` X gates and
`2h(w-1)` Toffolis. If the first Affine call has counts `(X1,C1,T1)` and the
final predicate `(X2,C2,T2)`, the complete clean MLP has

\[
X=2X_1+X_2+4h,\quad C=2C_1+C_2,\quad T=2T_1+T_2+2h(w-1).
\]

For `L` hidden layers, the same construction is applied inductively and reversed
in layer order. Let hidden layer `l` contain `h_l` words of width `w_l`, require
`a_l` arithmetic work qubits and have one-call counts `(X_l,C_l,T_l)`. If the
final predicate has `(X_f,C_f,T_f)` and requires `a_f` work, then

\[
Q=n_{\mathrm{in}}+1+2\sum_l h_lw_l+\max(a_1,\ldots,a_L,a_f),
\]

\[
X=2\sum_lX_l+X_f+\sum_{l:w_l>1}4h_l,
\]

\[
C=2\sum_lC_l+C_f,
\qquad
T=2\sum_lT_l+T_f+2\sum_lh_l\max(0,w_l-1).
\]

The maximum-work term is achieved by reusing one clean arithmetic region after
every layer.

For the single-record gradient channel, let `q` be the gradient-component width.
A signed variable-product compute uses `3q^2+q` Toffolis and `4q^2` CNOTs; clean
product copy and reverse therefore use `6q^2+2q` Toffolis and `8q^2+q` CNOTs per
feature. A clean affine residual oracle is applied, the residual and all products
are copied into the gradient output, and every operation is reversed. A full-word
equality tree then yields the phase predicate for the released gradient.

These results establish polynomial-size coherent access for the declared integer
architectures and the first actual training-leakage map. They do not establish
that the resulting fault-tolerant cost is lower than classical reconstruction.

## Single-record gradient no-advantage theorem

For exact squared-loss gradients of one biased linear record,

\[
g_w=(w^Tx+b-t)x,\qquad g_b=w^Tx+b-t.
\]

If `g_b != 0`, the exact decoder

\[
x_i=(g_w)_i/g_b,\qquad t=w^Tx+b-g_b
\]

recovers the candidate in linear time. If `g_b=0`, every representable
`(x,t=w^Tx+b)` belongs to the all-zero observation fibre. Consequently this task
is either classically trivial or information-theoretically ambiguous. The
compiled Grover path is a coherent-circuit verification artifact, not an
advantage claim.

## Acceptance gate for an end-to-end advantage claim

All conditions below must hold simultaneously:

1. the target equivalence relation and candidate prior are fixed before evaluation;
2. the coherent verifier is bit-level equivalent to a public reference evaluator;
3. all work qubits are uncomputed and phase marking is correct;
4. the number of marked candidates and non-equivalent collisions is reported;
5. oracle precision yields a non-vacuous hybrid success lower bound;
6. compiler, state preparation, inverse calls, shots, measurement, and readout are counted;
7. the classical baseline uses the same verifier, target success, and cost unit;
8. the measured parameter region satisfies the strict break-even inequality;
9. the structure-preserving compiler beats both exponential finite baselines on the reported scaling regime;
10. the verifier represents an actual reconstruction observation/objective rather than only a toy classifier output;
11. any original-sample recovery claim is below the applicable Bayes ceiling and respects all collision theorems;
12. the task is not already dominated by an analytic or specialized classical decoder;
13. all theorem assumptions, failure cases, random seeds, intervals and resource-conversion assumptions are released.
