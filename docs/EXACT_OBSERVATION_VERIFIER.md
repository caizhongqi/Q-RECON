# Structure-Preserving Exact-Observation Verifier

## 1. Reconstruction role

A model value oracle alone does not define a reconstruction search problem. Given
a public quantized observation word `t`, Q-RECON now compiles the exact predicate

\[
v_t(x)=\mathbf 1[F(x)=t]
\]

without enumerating the candidate truth table. The resulting circuit marks the
complete observation fibre

\[
F_t=\{x:F(x)=t\}.
\]

If this fibre contains multiple non-equivalent candidates, Grover search can
find a feasible candidate but cannot identify the original record without
additional information. The compiler therefore reports the marked-set size and
must be interpreted together with the Bayes/fibre bounds.

## 2. Clean equality comparator

Let `z` be a `q`-bit source word, `t` a fixed public constant, and `a` a clean
work register. Bits for which `t` is zero are temporarily inverted, reducing
comparison with `t` to an AND of all transformed source bits.

For `q>=3`, a forward Toffoli chain computes prefix conjunctions into `q-2`
clean work bits, one final Toffoli toggles the predicate target, and the prefix
chain is reversed. The temporary source inversions are then restored. Width one
uses one CNOT and width two uses one Toffoli.

### Theorem 1 — clean constant equality

For every source word `z`, target bit `y`, and clean work register,

\[
U_{=t}|z\rangle|y\rangle|0^a\rangle
=
|z\rangle|y\oplus\mathbf 1[z=t]\rangle|0^a\rangle.
\]

### Proof

After the constant-dependent X gates, every source control is one exactly when
the original word equals `t`. The conjunction chain therefore toggles the target
only in that case. Reversing the prefix Toffolis clears every conjunction work
bit while preserving the target, and reversing the X gates restores the source.
All gates are reversible basis permutations. \(\square\)

For width `q>=2`, the comparator uses exactly

\[
2q-3
\]

Toffoli gates, `q-2` reusable clean work bits, and

\[
2(q-\operatorname{wt}(t))
\]

X gates. Width one uses one CNOT and the same constant-dependent X count.

## 3. Composition with the affine value oracle

Let `U_F` be the clean structure-preserving affine value oracle. The exact
observation verifier performs:

1. `U_F` to materialize `F(x)` in an internal value register;
2. the clean equality comparator against `t`;
3. `U_F^{-1}` to clear the complete model value and arithmetic work.

### Theorem 2 — clean exact-observation predicate

The composed circuit satisfies

\[
U_v|x\rangle|y\rangle|0^a\rangle
=
|x\rangle|y\oplus\mathbf 1[F(x)=t]\rangle|0^a\rangle.
\]

The proof follows immediately from clean correctness of `U_F`, Theorem 1, and
Bennett uncomputation. Preparing the predicate target in `|-\rangle` yields the
phase oracle required by amplitude amplification.

If the affine value circuit has X/CNOT/Toffoli counts `(X_F,C_F,T_F)`, the
composed equality verifier has

\[
X=2X_F+2(q-\operatorname{wt}(t)),
\]

\[
C=2C_F+\mathbf 1[q=1],
\]

and

\[
T=2T_F+\begin{cases}
0,&q=1,\\
2q-3,&q\ge2.
\end{cases}
\]

Its additional comparator work is `max(0,q-2)` clean bits.

## 4. Exhaustive validation

The regression suite checks every source word, every target constant, and both
initial predicate-target states for widths one through five. It verifies source
preservation, exact equality, clean conjunction work, and inverse restoration.

For compiled affine models, all candidate inputs are enumerated and three values
are compared:

1. direct bit-exact affine evaluation;
2. the declared exact-observation predicate;
3. execution of the complete reversible gate circuit.

Grover simulation applies phases through the composed circuit and measures
success on the independently evaluated fibre.

## 5. Claim boundary

Exact output matching is not automatically exact training-record recovery. If
`|F_t|>1`, the verifier deliberately marks every colliding candidate. A paper
must report:

- the target observation and quantization;
- candidate prior and target equivalence relation;
- fibre size and non-equivalent collisions;
- original-record success separately from feasible-set success;
- classical and quantum costs at the same success event.

The equality verifier closes a necessary systems gap: the coherent search
predicate now evaluates a model-derived observation rather than a hand-written
answer bit. It does not remove the information-theoretic collision ceiling.
