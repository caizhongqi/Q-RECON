# Specialized Classical Baselines for Aggregate-Gradient Reconstruction

## 1. Why unstructured search is not the correct classical baseline

For a batch whose released observation is an additive sum of per-record
contributions,

\[
G(D)=\sum_{j=1}^{B} h_j(u_j),
\]

naive enumeration treats the Cartesian product
\(D=(u_1,\ldots,u_B)\) as an unstructured database. If every record position has
\(M\) local candidates, that baseline costs \(M^B\) evaluations. Comparing
Grover's \(O(M^{B/2})\) oracle calls only with this naive baseline would overstate
the quantum gain because the additive structure is classically exploitable.

Q-RECON therefore includes an exact balanced meet-in-the-middle solver and makes
it a mandatory baseline for every additive batch-gradient experiment.

## 2. Exact meet-in-the-middle algorithm

Choose a split after record \(s=\lfloor B/2\rfloor\). Enumerate all left partial
assignments and hash their vector sums:

\[
L(a)=\sum_{j=1}^{s}h_j(a_j).
\]

Enumerate every right partial assignment and compute

\[
R(b)=\sum_{j=s+1}^{B}h_j(b_j).
\]

A complete batch is a solution exactly when the hash table contains

\[
L(a)=G^*-R(b).
\]

All matching left assignments are combined with the right assignment, so the
algorithm returns the complete ordered observation fibre rather than only one
candidate. Every returned batch is rechecked by the public aggregate-gradient
evaluator.

### Theorem 1 — correctness

The meet-in-the-middle solver returns exactly the ordered candidates
\((u_1,\ldots,u_B)\) satisfying

\[
\sum_{j=1}^{B}h_j(u_j)=G^*.
\]

### Proof

If the solver emits a pair \((a,b)\), the hash lookup guarantees
\(L(a)=G^*-R(b)\), hence \(L(a)+R(b)=G^*\). Conversely, every satisfying complete
assignment has a left prefix and right suffix obeying the same equality. The left
sum is inserted during the complete left enumeration, so the corresponding right
lookup retrieves that prefix. Therefore no solution is omitted and no false
solution is accepted. \(\square\)

## 3. Complexity theorem

Let local domain sizes be \(M_1,\ldots,M_B\). The implementation enumerates

\[
N_L=\prod_{j=1}^{s}M_j,
\qquad
N_R=\prod_{j=s+1}^{B}M_j
\]

partial assignments. Hash construction and lookup take expected

\[
O(N_L+N_R+Z)
\]

time, where \(Z\) is the number of emitted solutions, and

\[
O(N_L)
\]

memory before optional left/right orientation optimization.

For equal domains of size \(M\), a balanced split gives

\[
O\!\left(M^{\lfloor B/2\rfloor}+M^{\lceil B/2\rceil}+Z\right)
=O\!\left(M^{\lceil B/2\rceil}+Z\right).
\]

The full candidate population is \(N=M^B\), so ideal unstructured Grover search
uses \(\Theta(\sqrt N)=\Theta(M^{B/2})\) oracle calls when the marked fraction is
constant-order appropriate.

### Corollary 1 — even batch size

For even \(B\), balanced meet-in-the-middle and ideal Grover search have the same
exponent:

\[
M^{B/2}.
\]

Grover may change constants or memory requirements, but there is no asymptotic
time-exponent separation over this classical baseline merely from treating the
additive batch as unstructured.

### Corollary 2 — two-record benchmark

For \(B=2\), meet-in-the-middle enumerates \(2M\) partial records, while the full
space contains \(M^2\) batches and Grover uses \(\Theta(M)\) oracle calls. The
current public-target 16-candidate benchmark has \(M=4\): the exact classical
solver examines eight partial states and recovers the unique batch. It is
therefore a coherent-compiler validation case, not an asymptotic quantum-
advantage example.

### Odd batch size

For odd \(B\), the simple balanced solver costs
\(O(M^{(B+1)/2})\), whereas ideal Grover scales as \(O(M^{B/2})\). This leaves at
most a square-root factor in the local-domain size before oracle construction,
fault-tolerant gates, state preparation, memory tradeoffs, and stronger classical
algorithms are counted. No advantage claim may be made from this expression alone.

## 4. Public and private target support

The solver constructs exact contribution tables from the same bit-level batch
specification used by the reversible oracle.

- With public targets, each local word contains only the record features and the
  position-specific public target is inserted into its contribution function.
- With private targets, each local word contains both features and target.

Signed words are decoded with the declared two's-complement width. Contributions
are accumulated as mathematical integers, and the released packed word is decoded
under the certified no-overflow contract. Thus the classical baseline and quantum
verifier solve exactly the same reconstruction problem.

## 5. Executable evidence

Regression tests verify:

- unique recovery of the 16-candidate public-target benchmark after only eight
  partial-state enumerations;
- equality between the complete private-target meet-in-the-middle fibre and brute
  force over all 256 ordered batches;
- correctness on a generic three-position vector-sum problem;
- equality between per-position contribution tables and the public batch-gradient
  evaluator;
- the even-batch exponent identity between balanced meet-in-the-middle and ideal
  unstructured Grover scaling.

The result object records split position, local domain sizes, enumerated left and
right states, hash-bucket count, complete solution count, optional truncation, and
all reconstructed candidate words.

## 6. Consequence for CCF-A-level claims

A top-tier evaluation must compare the compiled quantum search against the
strongest structure-aware classical baseline, not only exhaustive search. For
additive aggregate gradients, meet-in-the-middle is the minimum required baseline.
Depending on the candidate algebra, the evaluation must also consider generalized
birthday methods, algebraic elimination, lattice techniques, SAT/SMT, mixed-
integer optimization, branch-and-bound, and learned priors.

A credible quantum advantage region must survive all of the following:

1. the same candidate prior, target equivalence and success probability;
2. the meet-in-the-middle or stronger classical algorithm;
3. memory-time tradeoffs reported for both sides;
4. coherent verifier construction and inverse calls;
5. state preparation, fault-tolerant T cost, measurement and readout;
6. collisions and the applicable Bayes ceiling;
7. amortization of compiler setup over the declared workload.

The current result is deliberately a negative filter: it prevents a quadratic
query claim from being presented as a new end-to-end speedup when classical
additive structure already supplies the same exponent.
