# Non-Circular Coherent-Oracle Construction

## 1. The oracle-construction problem

A query-complexity statement begins after an oracle exists. A systems or
end-to-end claim cannot begin there: it must explain how the oracle is produced
from the released model, leakage transcript, and public reconstruction target.

This distinction is critical for reconstruction. A compiler that first evaluates
every candidate and records which candidates are marked may already have solved
the inverse problem before Grover search starts.

Let

\[
v_y:\{0,1\}^n\to\{0,1\}
\]

be a reconstruction verifier for observation `y`, with marked set

\[
M_y=\{x:v_y(x)=1\},\qquad |M_y|=K.
\]

The candidate population is `N=2^n`.

## 2. Explicit-table construction

A complete truth-table artifact is

\[
T_y=(v_y(0),v_y(1),\ldots,v_y(N-1)).
\]

It contains exactly the information needed to construct the inverse index

\[
I_y(1)=M_y,
\qquad
I_y(0)=\{0,1\}^n\setminus M_y.
\]

### Theorem 1 — preimage disclosure

Given the explicit table `T_y`, the complete marked set `M_y` can be recovered
with one linear scan using `N` table inspections and `O(K)` output space. If
`K=1`, the unique reconstruction answer is therefore recoverable classically
from the compiler artifact without making any calls to the compiled quantum
oracle.

#### Proof

Enumerate table positions `x=0,...,N-1` and append `x` exactly when `T_y[x]=1`.
The resulting list is, by definition, `M_y`. For `K=1`, its only element is the
unique answer. ∎

The executable counterpart is `build_truth_table_preimage_index`.

### Corollary 1 — minterm answer disclosure

For a one-bit minterm compiler, every marked input appears directly as the
`required_input` field of at least one emitted minterm-controlled X gate. When
`K=1`, reading that control pattern returns the answer.

This does not make minterm synthesis incorrect. It makes a one-shot speedup claim
circular if compiler setup and artifact inspection are excluded.

## 3. Enumerative setup versus Grover queries

`TruthTableOracle.from_function` evaluates the reference predicate on all `N`
candidates. Standard Grover search uses `Theta(sqrt(N/K))` coherent verifier
calls after compilation.

### Proposition 2 — single-instance enumerative setup barrier

If a verifier compiler materializes all `N` predicate values separately for the
same reconstruction instance, then its candidate-evaluation setup cost is
`Omega(N)`. For constant `K`, this asymptotically exceeds the ideal Grover query
count `Theta(sqrt(N/K))`.

Therefore

\[
C_{\mathrm{total,Q}}
=
C_{\mathrm{compile}}+C_{\mathrm{search,Q}}
\]

cannot be justified by quoting only `C_search,Q`.

This proposition concerns an enumerative compiler, not every possible coherent
oracle implementation. A structure-preserving arithmetic compiler can avoid the
`N` reference evaluations.

## 4. Reuse does not rescue a stored value table automatically

Suppose a full value table for

\[
f:\{0,1\}^n\to\{0,1\}^m
\]

is compiled once and reused for many target outputs `y`. A classical process can
construct

\[
I(y)=\{x:f(x)=y\}
\]

for every observed output in one `O(N)` pass, using a hash map keyed by `f(x)`.
Subsequent reconstruction requests are answered by table lookup in expected
`O(1+|I(y)|)` time.

Thus amortization must compare:

- quantum compilation plus repeated coherent searches; and
- classical table construction plus the equally reusable inverse index.

Reusing a table while forbidding the classical baseline from indexing the same
table is not a matched comparison.

## 5. ANF compression does not erase enumerative construction

The algebraic-normal-form backend can sharply reduce gate counts for Boolean
functions such as parity. In the current implementation, however, ANF
coefficients are obtained by a Möbius transform of all `2^n` truth-table values,
and the `ANFOracle` object retains the original table.

### Proposition 3 — gate compression is not setup compression

For the current truth-table-to-ANF pipeline, a smaller controlled-X/Toffoli
netlist does not reduce:

- the `N` reference evaluations needed to materialize the input table;
- the `Omega(N)` table payload;
- the availability of a complete classical preimage index.

Consequently, ANF is a valid exact circuit optimization baseline but not evidence
of non-circular end-to-end quantum advantage.

## 6. Structure-preserving compilation

A compiler is **candidate-enumeration-free** for a model family when its gate
construction loops over model dimensions, layers, coefficients, and word bits,
but never evaluates the verifier separately on all candidate assignments.

The current Q-RECON structure-preserving families include:

- integer affine value and threshold oracles;
- affine equality oracles;
- integer ReLU MLP and deep-MLP value oracles;
- exact single-record full-gradient equality oracles;
- fixed-point requantization and identity affine value oracles.

Their emitted circuits are polynomial in architecture dimensions and word widths.
They do not store a complete candidate-to-output table.

### Definition — non-circularity certificate

For the limited purpose of oracle construction, a compiler receives a
non-circularity certificate when:

1. candidate-space materialization is zero;
2. compiler reference evaluations do not scale as `2^n`;
3. the artifact does not contain a complete preimage table;
4. resource counts are derived from emitted gates, including uncomputation.

`audit_structure_preserving_oracle` records this certificate.

### Scope warning

Non-circular construction is necessary but not sufficient for a quantum
advantage claim. A structure-preserving predicate may still be easy to invert
classically. The batch-one biased-linear gradient example is explicit:

\[
x_i=\frac{g_{W_i}}{g_b}
\]

when `g_b` is nonzero. Q-RECON therefore treats the structure-preserving gradient
oracle as a correctness and cost benchmark, not as evidence that Grover search
beats the analytic attack.

## 7. Revised end-to-end cost equation

For reconstruction instance `y`, the minimum valid accounting is

\[
C_Q(y)=
C_{\mathrm{compile,Q}}(y)
+C_{\mathrm{prepare}}(y)
+q_Q C_{\mathrm{oracle,Q}}(y)
+C_{\mathrm{diffusion}}(y)
+C_{\mathrm{measure}}(y)
+C_{\mathrm{decode}}(y),
\]

and

\[
C_C(y)=
C_{\mathrm{preprocess,C}}(y)
+C_{\mathrm{search/invert,C}}(y).
\]

If the quantum compiler creates any reusable classical artifact, the classical
baseline receives the same artifact and may index, simplify, or invert it.

For multiple instances, setup can be amortized only when the setup artifact is
identical across those instances. Observation-dependent verifier compilation
cannot be amortized across different observations without explicitly showing how
the target dependence is factored out.

## 8. Executable audit fields

`OracleConstructionAudit` reports:

- candidate bits and `2^n` population;
- materialized truth-table entries;
- reference evaluations required by the enumerative pipeline;
- a lower bound on stored table payload bits;
- whether the artifact contains complete preimage information;
- marked count when directly available;
- whether a unique answer is recoverable from the artifact;
- ideal first-peak Grover iterations;
- whether enumerative setup exceeds those iterations;
- controlled-X terms and Toffoli gates;
- a claim-boundary statement.

The audit intentionally separates gate count from compiler information leakage.
Both must be reported.

## 9. Publication claim rule

A Q-RECON paper may use truth-table and ANF compilers for:

- exhaustive correctness checks;
- finite identifiability and collision enumeration;
- exact small-instance Grover simulation;
- resource upper bounds and synthesis comparisons.

It must not use them as the sole basis for end-to-end quantum advantage when the
compilation artifact already reveals the marked set or required `Theta(N)`
reference evaluations.

An end-to-end advantage claim requires all of the following:

1. candidate-enumeration-free coherent compilation;
2. no explicit preimage index in the artifact;
3. compiler cost included in the quantum total;
4. the same artifact and preprocessing rights for the classical baseline;
5. a lower bound, reduction, or strong empirical evidence against
   structure-aware classical inversion;
6. matched success probability and target equivalence;
7. a nonempty measured break-even region.

This rule removes a common source of apparent but circular quantum speedups.
