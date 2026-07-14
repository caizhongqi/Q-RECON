# Arbitrary-Depth Reversible ReLU MLP Oracle

## 1. Compiler milestone

The structure-preserving backend now supports an integer ReLU network with any
positive number of hidden affine layers followed by one affine threshold output:

\[
h_0=x,
\qquad
h_\ell=\operatorname{ReLU}(W_\ell h_{\ell-1}+b_\ell),
\quad 1\le\ell\le L,
\]

\[
v(x)=\mathbf 1[W_{L+1}h_L+b_{L+1}\ge\tau].
\]

Every layer is compiled to an explicit X/CNOT/Toffoli gate netlist. Hidden
preactivations and activations are materialized as finite two's-complement words;
all arithmetic and hidden registers return to zero after the predicate target is
toggled.

The current contract requires integer scales, signed hidden/output words,
componentwise ReLU, a one-bit threshold output, and interval-certified no-overflow
arithmetic. Fractional requantization remains a separate compiler milestone.

## 2. Forward compute and reverse cleanup

For each hidden layer, the compiler performs:

1. remap a clean structure-preserving affine value oracle onto the current input
   or previous activation registers;
2. compute its signed preactivation words;
3. copy componentwise ReLU values into clean activation registers using the exact
   sign-controlled construction;
4. reuse the same clean arithmetic work region for the next layer.

After the final affine-threshold oracle toggles the predicate qubit, layers are
uncomputed in reverse order. For layer \(\ell\), its ReLU copy is reversed first,
then its affine value oracle is reversed. The preceding activation registers are
still available at that point, so the inverse has exactly the inputs needed to
clear the layer.

### Theorem — clean arbitrary-depth MLP oracle

For every valid input word \(x\), target bit \(y\), and clean work register,

\[
U_{\mathrm{deep}}|x\rangle|y\rangle|0^a\rangle
=
|x\rangle|y\oplus v(x)\rangle|0^a\rangle.
\]

### Proof

Proceed through the hidden layers by induction. Before layer \(\ell\), all
activation registers through \(h_{\ell-1}\) contain their reference values and
the shared arithmetic work region is zero. Clean correctness of the affine
suboracle writes the exact preactivation and restores the shared work. The ReLU
copy writes exactly \(h_\ell\), preserves the preactivation, and restores its
sign control. Thus the induction invariant holds for the next layer.

The final clean predicate oracle toggles only the target and restores the shared
work. In reverse order, each ReLU inverse clears \(h_\ell\) while its
preactivation remains present, and the corresponding affine inverse clears that
preactivation while \(h_{\ell-1}\) remains present. The process ends with all
hidden and arithmetic registers in zero and the original input unchanged.
Every gate is reversible, so the basis-state identity extends linearly to
superpositions. \(\square\)

## 3. Shared-work liveness theorem

Let hidden layer \(\ell\) have \(h_\ell\) neurons of word width \(w_\ell\), and
let its clean affine oracle require \(a_\ell\) arithmetic work qubits. Let the
final predicate require \(a_f\) work qubits.

Because each affine suboracle returns its work to zero before the next layer, a
single region of size

\[
a_{\max}=\max\{a_1,\ldots,a_L,a_f\}
\]

is sufficient for the entire network. The total logical-qubit requirement is

\[
Q=n_{\mathrm{in}}+1+2\sum_{\ell=1}^{L}h_\ell w_\ell+a_{\max}.
\]

The two hidden-word terms are preactivation and activation storage. A naive
composition that allocates independent arithmetic work per layer would instead
pay \(\sum_\ell a_\ell+a_f\); the implementation and tests verify that the
compiler uses the maximum, not the sum.

## 4. Exact gate-count composition

Let \((X_\ell,C_\ell,T_\ell)\) be X/CNOT/Toffoli counts for one clean affine
value-oracle call at hidden layer \(\ell\), and let
\((X_f,C_f,T_f)\) be the final predicate counts. The complete clean deep MLP has

\[
X=2\sum_{\ell=1}^{L}X_\ell+X_f
+\sum_{\ell:w_\ell>1}4h_\ell,
\]

\[
C=2\sum_{\ell=1}^{L}C_\ell+C_f,
\]

\[
T=2\sum_{\ell=1}^{L}T_\ell+T_f
+2\sum_{\ell=1}^{L}h_\ell\max(0,w_\ell-1).
\]

Each hidden affine and ReLU block appears once in the forward computation and
once in reverse cleanup. The final predicate appears once. Under the repository's
exact-Toffoli accounting convention,

\[
T\text{-count}\le7T,
\qquad
T\text{-depth}\le3T.
\]

These are exact composition identities for the emitted netlist plus conservative
fault-tolerant conversion bounds. They are not optimal-synthesis claims.

## 5. Executable evidence

Small networks are exhaustively verified over every input candidate and both
initial target bits. The regression suite checks:

- equality with the public multi-layer `QuantizedNetwork` evaluator;
- every hidden activation at every layer;
- target XOR behavior and phase signs;
- complete ancilla cleanup and inverse restoration;
- the shared-work maximum-versus-sum identity;
- exact gate-count composition;
- the Grover success curve driven by the compiled phase netlist;
- rejection of unsupported fractional requantization.

## 6. Publication boundary

This result removes the fixed-depth limitation and gives a polynomial-size clean
oracle for arbitrary-depth integer ReLU MLPs. It still does not establish
end-to-end quantum advantage. A paper-level advantage claim additionally needs a
training-data reconstruction verifier, fixed-point precision/error analysis,
matched strongest classical baselines, state-preparation and readout accounting,
and a nonempty measured break-even region at the same success target.
