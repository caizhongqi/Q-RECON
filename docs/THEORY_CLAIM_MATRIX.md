# Q-RECON Theory Claim Matrix

The complete access-model, information, query, compiler, and cost arguments live
in the following documents:

- [`THEORY_FOUNDATIONS.md`](THEORY_FOUNDATIONS.md);
- [`BATCH_GRADIENT_NONIDENTIFIABILITY.md`](BATCH_GRADIENT_NONIDENTIFIABILITY.md);
- [`GRADIENT_RECONSTRUCTION_ORACLE.md`](GRADIENT_RECONSTRUCTION_ORACLE.md) and
  [`BATCH_GRADIENT_ORACLE.md`](BATCH_GRADIENT_ORACLE.md);
- [`COHERENT_ORACLE_SPEC.md`](COHERENT_ORACLE_SPEC.md);
- [`TRUTH_TABLE_ORACLE_BASELINE.md`](TRUTH_TABLE_ORACLE_BASELINE.md) and
  [`ANF_ORACLE_OPTIMIZATION.md`](ANF_ORACLE_OPTIMIZATION.md);
- [`STRUCTURE_PRESERVING_AFFINE_ORACLE.md`](STRUCTURE_PRESERVING_AFFINE_ORACLE.md),
  [`REVERSIBLE_MLP_ORACLE.md`](REVERSIBLE_MLP_ORACLE.md), and
  [`DEEP_REVERSIBLE_MLP_ORACLE.md`](DEEP_REVERSIBLE_MLP_ORACLE.md);
- [`FIXED_POINT_REVERSIBLE_COMPILATION.md`](FIXED_POINT_REVERSIBLE_COMPILATION.md);
- [`ORACLE_CONSTRUCTION_NONCIRCULARITY.md`](ORACLE_CONSTRUCTION_NONCIRCULARITY.md);
- [`END_TO_END_COST_PROTOCOL.md`](END_TO_END_COST_PROTOCOL.md); and
- [`THEORY_EVALUATION_PROTOCOL.md`](THEORY_EVALUATION_PROTOCOL.md).

## Claim ledger

| Claim | Mathematical status | Executable evidence | Publication status |
|---|---|---|---|
| deterministic observation fibres impose an exact Bayes ceiling | proved for finite candidate spaces | `bayes_reconstruction_success` | ready as a foundational lemma, not a novelty claim |
| recovery modulo a declared equivalence relation has a class-posterior Bayes optimum | proved for deterministic and noisy channels | `bayes_equivalence_reconstruction_success`, `channel_bayes_equivalence_reconstruction_success` | ready as a formal evaluation rule |
| stochastic post-processing cannot increase exact guessing probability | proved by data processing | `postprocess_channel` plus tests | ready as a foundational lemma |
| full-column gradient Jacobian rank certifies local injectivity | standard inverse-function argument | `gradient_jacobian_report.full_column_rank` | ready only as a local certificate |
| rank deficiency proves non-identifiability | false in general | documented counterexample | must not be claimed |
| finite truth-table enumeration gives global collision fibres | exact for the enumerated candidate space | `analyze_finite_oracle` | ready as a finite-space certificate; it does not extrapolate beyond the domain |
| biased linear-regression aggregate gradients identify the complete batch for `B>=2` | false when inputs and regression targets are both private | explicit continuous `A 1 = 1` batch-mixing collision family and finite non-permutation collision fibres | non-identifiability theorem ready under its stated scope |
| the same batch-mixing theorem automatically covers fixed labels, softmax, or arbitrary networks | not proved | no supporting construction | must not be claimed |
| a fixed public-label aggregate-gradient channel may be injective on a bounded finite candidate space | true for the declared 16-candidate two-record benchmark | exhaustive 16/16 distinct observations, clean equality oracle and Grover regression | ready only as a finite-domain certificate, not a general theorem |
| finite public-label injectivity establishes quantum advantage | false | no classical lower bound; algebraic, meet-in-the-middle, SAT/SMT and integer-programming baselines remain | must not be claimed |
| a full exact single-record biased-linear squared-loss gradient is uniquely invertible when its bias gradient is nonzero | proved by `x_i=(g_w)_i/g_b` and `t=w^T x+b-g_b` | analytic decoder, finite fibre enumeration and exhaustive tests | ready as a Q-RECON-specific identifiability theorem |
| the zero-gradient single-record case identifies the original record | false whenever more than one representable `x` satisfies `t=w^T x+b` | explicit all-zero fibre enumeration | must not be claimed; the Bayes ceiling applies |
| Grover gives meaningful advantage for full exact single-record biased-linear gradients | false under the declared arithmetic model | nonzero case has an `O(d)` decoder; zero case is non-identifiable | negative no-advantage corollary ready |
| identical released quantum states cannot be distinguished | standard Helstrom result | `binary_helstrom_success` | ready as an information bound |
| a classical prediction API supplies coherent queries | false by access-model definition | access models documented | must not be claimed |
| standard Grover search gives a black-box quadratic query reduction | standard result under clean Q-Access | exact success/query helpers and tests | usable only with explicit oracle assumptions |
| mixed-polarity minterm synthesis implements a clean finite value oracle | proved for the declared truth-table construction | `TruthTableOracle`, gate-list execution and exhaustive basis-permutation tests | ready as a correctness theorem for the exponential baseline |
| ANF synthesis implements the same clean finite value oracle | proved by uniqueness of the GF(2) multilinear polynomial | `ANFOracle`, Möbius transform and exhaustive cross-backend tests | ready as an exact optimization theorem |
| ANF always beats minterm synthesis | false | `compare_exact_syntheses` retains both records and chooses by a declared resource key | must not be claimed |
| affine Boolean predicates can have zero Toffoli cost under ANF | proved from degree-at-most-one monomials | parity regression and scaling reports | ready under the stated decomposition convention |
| the finite verifier gives the declared phase sign through kickback | follows from clean one-bit XOR semantics | phase tests and Grover simulation | ready for the finite baseline |
| truth-table or truth-table-to-ANF compilation is a non-circular one-shot reconstruction pipeline | false when the full candidate table is materialized | `build_truth_table_preimage_index`, `audit_truth_table_oracle`, `audit_anf_oracle` | must not support an end-to-end advantage claim; the artifact can reveal the full marked set |
| minterm gates hide the unique marked answer from a classical reader | false for a one-bit unique verifier | the marked word is the minterm `required_input`; inverse-index tests recover it | must not be claimed |
| avoiding candidate enumeration proves classical inversion hardness | false | `audit_structure_preserving_oracle` certifies only absence of table circularity | a separate reduction, lower bound, or strongest-baseline study is required |
| an integer affine model can be compiled to a clean polynomial-size value or threshold oracle | proved for two's-complement, no-overflow, shift-add and ripple-carry semantics | `ReversibleIntegerAffineValueOracle`, `ReversibleIntegerAffinePredicateOracle`, exhaustive tests and exact resource formulas | ready as a structure-preserving compiler theorem |
| a two-layer integer `Affine-ReLU-Affine/Threshold` network can be compiled to a clean predicate and phase oracle | proved by clean composition, sign-controlled ReLU and Bennett uncomputation | `ReversibleIntegerMLPPredicateOracle`, exhaustive candidate/target/inverse tests and Grover regression | ready under the stated integer/two-layer scope |
| an arbitrary-depth integer ReLU MLP can be compiled with one shared arithmetic work region | proved by layerwise induction, reverse liveness cleanup and maximum-work reuse | `ReversibleIntegerDeepMLPPredicateOracle`, exhaustive deep tests and resource identities | ready under integer semantics |
| an exact single-record linear-training gradient can be compiled as a polynomial-size clean value/equality/phase oracle | proved by residual computation, signed variable multiplication, full-gradient comparison and reverse cleanup | `ReversibleSingleRecordGradientValueOracle`, `ReversibleSingleRecordGradientEqualityOracle`, exhaustive tests | ready as the first structure-preserving training-leakage compiler theorem |
| an ordered batch sum-gradient can be compiled with one reusable record-gradient and arithmetic work region | proved by clean record composition, modular accumulation, output copy and reverse cleanup | `ReversibleBatchGradientValueOracle`, `ReversibleBatchGradientEqualityOracle`, exhaustive tests and resource reports | ready as an aggregate-training-leakage compiler theorem |
| half-away-from-zero fixed-point downscaling has a clean polynomial-size reversible implementation | proved for overflow-free `f_target <= f_source`, signed/signed or unsigned/unsigned formats | `ReversibleFixedPointRequantizationOracle`, controlled-increment/negation primitives and exhaustive word tests | ready as a fixed-point arithmetic theorem under the declared scope |
| an identity fixed-point affine layer can be compiled bit-for-bit to the reference semantics | proved by product-scale integer accumulation, exact bias alignment, clean per-output requantization and reverse affine cleanup | `ReversibleFixedPointAffineValueOracle`, exhaustive multi-output tests and instantiated resources | ready as a fixed-point affine compiler theorem |
| arbitrary fixed-point upscaling, saturation and deep ReLU MLP lowering are supported | false | these paths are rejected or not yet composed | must not be claimed |
| the present VQC prior gives quantum query advantage | not established | no supporting coherent-oracle advantage experiment | must not be claimed |
| a structure-preserving compiled neural verifier preserves query advantage end to end | open | clean integer/fixed-point affine, equality, deep ReLU MLP, single-gradient and batch-gradient compilers plus cost planning and construction audits now exist | requires a harder identifiable leakage, strongest classical solvers, matched preprocessing, precision/state-preparation costs and a measured nonempty break-even region |
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
formula. The deterministic result follows by setting `W(y|x)=1[g(x)=y]`.

Exact-candidate and equivalence-class success must be reported separately. In
particular, ordered batch recovery and recovery modulo record permutation are
different tasks and have different Bayes ceilings.

## Exact finite compiler theorem and limitation

For a bit function `f:{0,1}^n -> {0,1}^m`, the mixed-polarity backend emits one
fully controlled X for every set output bit of every input minterm. If

\[
S=\sum_x \operatorname{wt}(f(x)),
\]

it emits exactly `S` controlled terms and, for `n>=2`, uses `S(2n-3)` Toffolis
under the declared clean-ancilla decomposition. The ANF backend emits the unique
multilinear GF(2) polynomial of each output bit. A degree-`d>=2` monomial uses
`2d-3` Toffolis; degree zero and one use X and CNOT only. Both satisfy

\[
U_f|x\rangle|y\rangle|0^a\rangle
=|x\rangle|y\oplus f(x)\rangle|0^a\rangle.
\]

These are correctness and finite-space cross-check backends. They are not valid
one-shot end-to-end speedup evidence when all `2^n` values are evaluated during
compilation. The same table supports a classical inverse index; for a unique
mark it returns the answer before quantum search begins.

## Structure-preserving compiler theorems

The integer affine backend computes constant products by sign/zero extension and
shift-add, accumulates with clean ripple-carry addition, copies the result and
reverses every arithmetic operation. Under a no-overflow proof, the modular bit
circuit equals the mathematical integer affine map.

For a two-layer MLP, the compiler composes a clean first affine value oracle, a
reversible componentwise ReLU copy and a final affine-threshold oracle, then
reverses ReLU and the first affine call. For hidden width `w>1`, `h` hidden
neurons, first-layer counts `(X1,C1,T1)` and final counts `(X2,C2,T2)`,

\[
X=2X_1+X_2+4h,\quad C=2C_1+C_2,\quad
T=2T_1+T_2+2h(w-1).
\]

For `L` hidden layers, the construction is applied inductively and reversed in
layer order. If hidden layer `l` has `h_l` words of width `w_l`, one-call counts
`(X_l,C_l,T_l)` and work `a_l`, while the final predicate has
`(X_f,C_f,T_f)` and work `a_f`, then

\[
Q=n_{\mathrm{in}}+1+2\sum_l h_lw_l+\max(a_1,\ldots,a_L,a_f),
\]

\[
X=2\sum_lX_l+X_f+\sum_{l:w_l>1}4h_l,
\]

\[
C=2\sum_lC_l+C_f,\qquad
T=2\sum_lT_l+T_f+2\sum_lh_l\max(0,w_l-1).
\]

The maximum-work term follows from reuse of one clean arithmetic region.

For a single-record gradient channel with component width `q`, a signed variable
product compute uses `3q^2+q` Toffolis and `4q^2` CNOTs. Product copy plus reverse
therefore uses `6q^2+2q` Toffolis and `8q^2+q` CNOTs per feature. For an ordered
batch of `B` records and `k=d+1` gradient components, if a clean record call has
counts `(Xr,Cr,Tr)` and one `q`-bit adder has `(Xa,Ca,Ta)`, the aggregate value
oracle has

\[
X=4BX_r+2BkX_a,
\]

\[
C=4BC_r+2BkC_a+kq,
\]

\[
T=4BT_r+2BkT_a,
\]

plus public-target constant-load X gates when applicable. Record work is reused,
so it is not multiplied by `B` in the peak-ancilla count.

## Fixed-point compiler theorem

Let a source code `q` carry `f_s` fractional bits and let `s=f_s-f_t>=0`. The
reference downscaler is

\[
R_s(q)=\operatorname{sgn}(q)
\left\lfloor\frac{|q|+2^{s-1}}{2^s}\right\rfloor
\]

for `s>0`, with `R_0(q)=q`. Conditional absolute value, half-unit addition,
logical shift/copy, sign restoration, output XOR and full uncomputation implement
this map exactly. If `m=max(n+1,s+1)` is the magnitude width and `t` the target
width, a signed instance obeys

\[
T_R\le 2\left((m-1)^2+2m\mathbf 1[s>0]+(t-1)^2\right).
\]

Composing the raw product-scale affine oracle with one clean requantizer per
output gives bit-for-bit `QuantizedAffineLayer.evaluate_codes` semantics for
identity activation and overflow-free ranges.

## Single-record gradient no-advantage theorem

For exact squared-loss gradients of one biased linear record,

\[
g_w=(w^Tx+b-t)x,\qquad g_b=w^Tx+b-t.
\]

If `g_b != 0`, the decoder

\[
x_i=(g_w)_i/g_b,\qquad t=w^Tx+b-g_b
\]

recovers the record in linear time. If `g_b=0`, every representable
`(x,t=w^Tx+b)` belongs to the all-zero fibre. This task is therefore either
classically trivial or information-theoretically ambiguous. Its compiled Grover
path is a coherent-circuit verification artifact, not an advantage claim.

## Aggregate-gradient claim boundary

Private-label aggregate gradients have explicit continuous and finite collision
families, including collisions beyond record permutation. Public labels can make
particular bounded domains injective; the current 16-candidate benchmark is one
exhaustively certified example. That certificate permits a unique phase mark but
supplies no classical time lower bound. Any future advantage result must compare
against algebraic elimination, meet-in-the-middle, SAT/SMT, mixed-integer and
optimized gradient-matching solvers on the same candidate prior.

## Acceptance gate for an end-to-end advantage claim

All conditions below must hold simultaneously:

1. the target equivalence relation and candidate prior are fixed before evaluation;
2. the coherent verifier is bit-level equivalent to a public reference evaluator;
3. all work qubits are uncomputed and phase marking is correct;
4. the number of marked candidates and non-equivalent collisions is reported;
5. oracle precision yields a non-vacuous hybrid success lower bound;
6. compiler, state preparation, inverse calls, shots, measurement and readout are counted;
7. the compiler does not enumerate the full candidate domain, or that enumeration is charged in full;
8. any compiler artifact and inverse-indexing opportunity are also given to the classical baseline;
9. the classical baseline uses the same verifier, target success and cost unit;
10. the measured parameter region satisfies the strict break-even inequality;
11. the structure-preserving compiler beats finite minterm/ANF baselines in the reported scaling regime;
12. the verifier represents a real reconstruction observation/objective, not only a toy classifier output;
13. any original-sample claim respects the applicable Bayes ceiling and collision theorems;
14. the task is not dominated by an analytic or specialized classical decoder;
15. ordered and permutation-equivalent batch success are both reported when relevant;
16. strongest algebraic, SAT/SMT, mixed-integer, meet-in-the-middle and optimized attack baselines are evaluated where applicable;
17. all theorem assumptions, failure cases, random seeds, confidence intervals and resource-conversion assumptions are released.
