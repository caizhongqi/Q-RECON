# Reversible Fixed-Point Requantization and Affine Compilation

## 1. Purpose

The first structure-preserving Q-RECON compilers operated on integer codes with
zero fractional bits. That was sufficient to certify clean value, verifier, and
phase-oracle mechanics, but it did not implement the fixed-point rounding rule
used by `QuantizedAffineLayer`.

This milestone closes that gap for **overflow-free downscaling and identity
fixed-point affine layers**. It provides a gate-level X/CNOT/Toffoli circuit,
not a truth-table replacement.

The implementation consists of:

- `ReversibleFixedPointRequantizationOracle`;
- `append_controlled_increment`;
- `append_controlled_twos_complement`;
- `ReversibleFixedPointAffineValueOracle`;
- `compile_structure_preserving_fixed_point_affine_oracle`.

## 2. Exact numerical contract

A source integer code `q` with `f_s` fractional bits represents

\[
q\,2^{-f_s}.
\]

For a target format with `f_t <= f_s`, let

\[
s=f_s-f_t.
\]

Q-RECON uses round-to-nearest with exact half ties away from zero:

\[
R_s(q)=
\begin{cases}
q,&s=0,\\[3pt]
\operatorname{sgn}(q)
\left\lfloor\dfrac{|q|+2^{s-1}}{2^s}\right\rfloor,&s>0.
\end{cases}
\]

This is bit-for-bit identical to `round_shift_right` and `rescale_code` in
`fixed_point.py`.

The current circuit deliberately rejects:

- fractional upscaling (`f_t > f_s`);
- any source range whose rounded values do not fit the target word;
- modular-overflow or saturation lowering;
- signed full-domain inputs mapped to an unsigned target.

Rejecting unsupported semantics is preferable to silently compiling a different
function.

## 3. Controlled increment primitive

For a clean control bit `c` and an `n`-bit little-endian register `z`, the
primitive implements

\[
|c\rangle|z\rangle|0\rangle
\longmapsto
|c\rangle|z+c\pmod{2^n}\rangle|0\rangle.
\]

The circuit visits target bits from most significant to least significant. Bit
`i>0` is toggled exactly when `c=1` and all lower pre-increment bits are one;
the least significant bit is toggled by a CNOT. Each multi-controlled X is
decomposed into a clean Toffoli chain and immediately uncomputed.

### Lemma 1 — controlled increment correctness

For every basis state `c,z`, `append_controlled_increment` preserves `c`, maps
`z` to `z+c mod 2^n`, and restores every work bit to zero.

#### Proof

When `c=0`, every controlled operation is disabled. When `c=1`, a bit changes
exactly if the carry generated at bit zero propagates through all lower one bits.
Processing from high to low ensures those controls still hold the original
pre-increment values. The final CNOT toggles bit zero. The Toffoli chains are
compute-use-uncompute constructions, so their work bits return to zero. This is
precisely binary addition by one modulo `2^n`. ∎

With the clean-chain decomposition used here, an `n`-bit controlled increment
uses at most

\[
(n-1)^2
\]

Toffoli gates and one CNOT on the least significant bit. The formula is exact
for the emitted decomposition when `n>=1`.

## 4. Conditional two's complement

Conditional negation first flips every register bit under control `c`, then
performs the controlled increment:

\[
|c\rangle|z\rangle
\mapsto
|c\rangle
\begin{cases}
|z\rangle,&c=0,\\
|-z\bmod 2^n\rangle,&c=1.
\end{cases}
\]

It therefore uses `n` additional CNOTs and the controlled-increment cost above.

## 5. Clean requantization oracle

Let `n` be the source width, `t` the target width, and

\[
m=\max\{n+1,s+1\}
\]

be the unsigned magnitude-work width. The circuit performs:

1. copy and sign-extend the source word into an `m`-bit work register;
2. conditionally take its absolute value;
3. add the half-unit constant `2^(s-1)` when `s>0`;
4. copy bits `s,...,s+t-1` into a quotient register;
5. conditionally restore the original sign in `t` bits;
6. XOR the quotient into the designated output;
7. reverse every preceding operation.

### Theorem 2 — bit-exact clean requantization

Assume every rounded source code fits the target format. For every source word
`x`, output word `y`, and zero-initialized work register,

\[
U_R|x\rangle|y\rangle|0^a\rangle
=
|x\rangle|y\oplus R_s(x)\rangle|0^a\rangle,
\]

where `R_s(x)` is encoded in the declared target two's-complement or unsigned
format.

#### Proof

For a nonnegative source, the magnitude register contains `q`. For a negative
source, sign extension followed by conditional two's complement contains
`|q|`. Adding `2^(s-1)` and copying from bit position `s` computes the exact
integer floor in the definition of `R_s`. Conditional two's complement of the
quotient restores the sign. The output copy is XOR, so arbitrary `y` is
supported. Reversing the complete compute network restores the magnitude,
quotient, constant-addend, carry, and multi-control work registers to zero while
leaving the copied output intact. ∎

### Resource bound

For signed source words, the exact emitted Toffoli count is bounded by

\[
T_R
\le
2\left((m-1)^2+2m\,\mathbf 1[s>0]+(t-1)^2\right).
\]

The outer factor two is compute/uncompute. The `2m` term is the clean CDKM
constant addition. For unsigned words the conditional-negation terms disappear.
Thus the requantizer is polynomial in the word widths:

\[
T_R=O(m^2+t^2).
\]

The implementation also reports instantiated X, CNOT, Toffoli, T-count,
T-depth, logical depth, logical qubits, and peak clean ancillas.

## 6. Fixed-point affine composition

For an affine layer

\[
z_j=\sum_i w_{ji}x_i+b_j,
\]

input and weight codes are multiplied at fractional precision

\[
f_p=f_x+f_w.
\]

Bias codes are deterministically aligned to `f_p`. Static interval analysis
chooses a signed accumulator width that contains every reachable raw sum. The
compiler then:

1. runs the existing clean integer affine value oracle at product scale;
2. retains each raw accumulator word in a dedicated register;
3. applies the clean fixed-point requantizer to every output, reusing one clean
   work region sequentially;
4. reverses the raw affine oracle.

### Theorem 3 — fixed-point affine reference equivalence

For an identity-activation `QuantizedAffineLayer` satisfying the no-overflow
range obligations, the compiled circuit implements

\[
U_L|x\rangle|y\rangle|0^a\rangle
=
|x\rangle|y\oplus L_b(x)\rangle|0^a\rangle,
\]

where `L_b` is exactly `QuantizedAffineLayer.evaluate_codes`, including bias
alignment and half-away-from-zero output requantization.

#### Proof

The raw affine subcircuit is reference-equivalent to integer accumulation at
scale `f_p`. Theorem 2 maps each reachable raw accumulator to the exact output
code used by the classical layer. Sequential work reuse is valid because every
requantizer call is clean. The final inverse affine call removes all raw outputs
and arithmetic work without changing the copied final output. ∎

If the raw affine resource is `C_aff` and the layer has `o` outputs, the total
circuit size satisfies

\[
C_{\mathrm{fp-affine}}
=2C_{\mathrm{aff}}+oC_R+O(o),
\]

under the actual composed implementation, and remains polynomial in dimensions
and bit widths. Instantiated gate counts remain the source of truth for concrete
experiments.

## 7. Exhaustive verification

`tests/test_requantization.py` verifies:

- every source word for multiple signed and unsigned formats;
- every output word for the standalone requantizer;
- positive and negative half-tie cases;
- controlled-increment correctness and inverse restoration;
- rejection of unsafe target ranges and unsupported upscaling;
- every candidate input for a two-output fixed-point affine layer;
- arbitrary output XOR semantics, inverse correctness, and zero final garbage;
- deterministic resource reports.

These tests certify the finite configurations under test. They do not replace a
formal proof for arbitrary widths; the theorem and circuit construction provide
that general argument.

## 8. Claim boundary and next milestone

This milestone removes the previous statement that reversible fixed-point
rounding was wholly unsupported. The supported claim is now narrower and
precise:

> Overflow-free fixed-point downscaling and identity affine layers with
> half-away-from-zero rounding have a clean structure-preserving reversible
> implementation and exhaustive small-width verification.

It does **not** yet establish:

- saturating arithmetic;
- fixed-point upscaling;
- a composed fixed-point ReLU MLP;
- CNN or transformer lowering;
- fault-tolerant wall-clock advantage;
- end-to-end quantum advantage over the strongest structure-aware classical
  reconstruction algorithm.

The immediate compiler extension is to compose fixed-point affine outputs with
clean sign/ReLU handling and then reuse the deep-MLP work-region scheduler under
per-layer fractional formats.
