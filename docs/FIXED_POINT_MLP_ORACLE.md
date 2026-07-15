# Fixed-Point ReLU MLP Coherent Oracle

## 1. Scope

This document extends the overflow-free fixed-point affine compiler from
`FIXED_POINT_REVERSIBLE_COMPILATION.md` to the first nonlinear fixed-point model
family supported by Q-RECON:

\[
x\longmapsto L_2(\operatorname{ReLU}(L_1(x))).
\]

The implementation provides three clean structure-preserving backends:

- `ReversibleFixedPointAffineReLUValueOracle`;
- `ReversibleFixedPointMLPValueOracle`;
- `ReversibleFixedPointMLPPredicateOracle`.

The hidden layer uses the exact fixed-point affine semantics already proved for
`ReversibleFixedPointAffineValueOracle`, followed by a bit-level two's-complement
ReLU. The output layer uses identity activation. A one-output model can be
converted into a clean threshold and phase oracle without enumerating the
candidate space.

## 2. Two's-complement ReLU copy

Let `z` be a signed `w`-bit two's-complement word with sign bit `s=z_(w-1)`.
ReLU is

\[
\operatorname{ReLU}(z)=
\begin{cases}
z,&s=0,\\
0,&s=1.
\end{cases}
\]

For every nonsign bit `j<w-1`, the compiler applies a Toffoli controlled on
`not s` and `z_j` with the designated output bit as target. A negative control is
implemented by applying X to `s` before the group of Toffolis and restoring X
afterwards. The output sign bit is never copied: it is zero for all nonnegative
inputs and ReLU outputs zero for negative inputs.

For unsigned fixed-point words, ReLU is the identity and the compiler uses one
CNOT per output bit.

### Lemma 1 — clean ReLU value copy

For every signed preactivation word `z`, output word `y`, and zero-initialized
work register,

\[
|z\rangle|y\rangle
\longmapsto
|z\rangle|y\oplus\operatorname{ReLU}(z)\rangle.
\]

The preactivation is unchanged at the end of the copy.

#### Proof

After the first X, the sign wire is one exactly for nonnegative inputs. Each
Toffoli therefore copies a nonsign bit iff the original input is nonnegative.
Negative inputs copy no bits; nonnegative inputs copy every information-bearing
bit. The second X restores the sign. The operation is self-inverse because all
emitted gates are self-inverse. ∎

For `h` signed output words of width `w`, the ReLU copy contributes exactly

\[
2h\quad\text{X gates},
\qquad
h(w-1)\quad\text{Toffoli gates}.
\]

The full affine/ReLU value oracle computes the fixed-point affine result into a
preactivation register, copies ReLU into the designated output, and reverses the
entire affine call. If one affine value call has counts `(X_A,C_A,T_A)`, the
composed value oracle has

\[
X=2X_A+2h,
\qquad
C=2C_A,
\qquad
T=2T_A+h(w-1).
\]

## 3. Two-layer fixed-point value oracle

Let the hidden layer `L_1` have ReLU activation and output format `F_h`. Let the
final identity layer `L_2` use exactly `F_h` as its input format. The compiler
allocates:

1. the public candidate input register;
2. the designated final output register;
3. one hidden-activation register;
4. clean work for the hidden affine/ReLU call;
5. clean work for the final affine call.

It executes:

1. `U_(L1,ReLU)` into the hidden register;
2. `U_L2` from the hidden register into the final output;
3. `U_(L1,ReLU)^dagger` to erase the hidden register and all hidden work.

The final affine call is itself clean, so its work is already zero before the
hidden inverse begins.

### Theorem 2 — fixed-point MLP reference equivalence

Assume:

- every fixed-point affine range proof is overflow-free;
- `L_1` uses ReLU and `L_2` uses identity activation;
- the hidden output and final input formats agree exactly;
- every requantization is a supported downscaling or equal-scale operation.

Then the compiled circuit satisfies

\[
U_M|x\rangle|y\rangle|0^a\rangle
=
|x\rangle
|y\oplus L_{2,b}(\operatorname{ReLU}(L_{1,b}(x)))\rangle
|0^a\rangle,
\]

where both layer maps are bit-for-bit the corresponding
`QuantizedAffineLayer.evaluate_codes` reference semantics.

#### Proof

The hidden affine subcall is reference-equivalent by the fixed-point affine
theorem. Lemma 1 copies exactly its componentwise ReLU into the hidden register
while preserving the preactivation until reverse cleanup. The final clean affine
call consumes the hidden register without changing it and XORs the correct final
word into the output. Reversing the hidden call removes every hidden activation,
preactivation, arithmetic and requantization work bit. ∎

If hidden and output value calls have gate counts `(X_h,C_h,T_h)` and
`(X_o,C_o,T_o)`, the composed value oracle emits

\[
X=2X_h+X_o,
\qquad
C=2C_h+C_o,
\qquad
T=2T_h+T_o.
\]

These are exact sequential-composition counts for the emitted circuit. Peak
qubits are the candidate and final-output registers plus the hidden activation
and both child work regions. Future optimization may reuse child work regions
when their liveness intervals do not overlap.

## 4. Threshold and phase oracle

For a single final output code `q(x)` and a public representable threshold
`tau`, define

\[
v_\tau(x)=\mathbf 1[q(x)\ge\tau].
\]

The predicate compiler:

1. computes the clean fixed-point MLP value into a temporary logit register;
2. applies a structure-preserving signed or unsigned integer affine-threshold
   predicate to that register;
3. reverses the complete value oracle.

An accumulator one bit wider than the final word represents every difference
`q-tau` without overflow. The resulting clean predicate satisfies

\[
U_v|x\rangle|z\rangle|0^a\rangle
=
|x\rangle|z\oplus v_\tau(x)\rangle|0^a\rangle.
\]

Preparing the target in `|->` gives

\[
O_v|x\rangle=(-1)^{v_\tau(x)}|x\rangle.
\]

### Theorem 3 — phase correctness

For every valid candidate basis word, `phase_sign(x)` equals `-1` exactly when
the bit-exact fixed-point reference model returns a code at least `tau`.

#### Proof

Theorem 2 establishes the temporary logit word. The existing clean integer
affine-threshold theorem establishes comparison against `tau` in one widened
signed accumulator. Reversing the value call erases the logit and its work
without changing the copied predicate target. Standard kickback then yields the
stated phase. ∎

## 5. Candidate-enumeration freedom

The compiler loops over:

- affine rows and features;
- coefficient bits;
- arithmetic word bits;
- hidden neurons and output neurons.

It does not evaluate the network on all `2^n` candidate assignments and does not
store a candidate-to-label table. It therefore satisfies the limited
non-circular construction certificate in
`ORACLE_CONSTRUCTION_NONCIRCULARITY.md`.

This statement does **not** prove that the fixed-point MLP is classically hard to
invert. A valid advantage result still requires the strongest structure-aware
classical solvers and matched preprocessing rights.

## 6. Exhaustive verification

`tests/test_fixed_point_mlp.py` checks small but complete domains:

- hidden affine/ReLU output equals the classical reference for every candidate;
- all negative preactivations map to zero;
- arbitrary output words receive XOR rather than overwrite semantics;
- the two-layer value oracle and its inverse return every work bit to zero;
- the threshold oracle matches every reference label;
- phase signs match the truth table;
- the compiled predicate drives the standard Grover success curve;
- incompatible activations, dimensions, formats and thresholds are rejected;
- resource reports are deterministic and nonzero for nontrivial circuits.

These finite tests are independent checks of the general circuit argument. They
are not the source of the polynomial compiler.

## 7. Claim boundary

The supported claim is now:

> Overflow-free two-layer fixed-point `Affine-ReLU-Affine` models with exact
> half-away-from-zero downscaling have clean structure-preserving value,
> threshold and phase oracles.

The implementation does not yet establish:

- arbitrary-depth fixed-point MLP composition;
- saturating or wraparound arithmetic;
- fractional upscaling;
- softmax, convolution, attention or normalization;
- a lower bound against specialized classical inversion;
- end-to-end fault-tolerant quantum advantage.

Those are separate milestones and must not be inferred from this compiler
correctness result.
