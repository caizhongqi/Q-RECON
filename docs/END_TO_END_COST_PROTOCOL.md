# Matched-Success End-to-End Search Cost Protocol

## 1. Purpose

A reduction in oracle-query count is not an end-to-end advantage. Q-RECON uses
this protocol to compare classical and coherent reconstruction search only after
both methods are required to reach the same target success probability and all
reported costs are expressed in one declared unit.

The implementation is a sensitivity-analysis engine. A numerical advantage is a
conditional statement under the supplied gate, loading, compilation, and
classical-evaluation prices; it is not automatically a wall-clock or hardware
claim.

## 2. Search instance

Let a clean predicate oracle define a candidate population of size

\[
N=2^n
\]

with exactly `K` marked candidates. The target event is recovery of any declared
acceptable marked candidate. Original-sample recovery requires the marked set to
coincide with the stated target equivalence class; otherwise the search event and
the reconstruction event must be reported separately.

Fix a required success probability

\[
p_{\mathrm{target}}\in(0,1].
\]

## 3. Classical matched-success plan

For uniform distinct queries without replacement, the exact success after `q`
queries is

\[
P_C(q)=1-\frac{\binom{N-K}{q}}{\binom{N}{q}}.
\]

The implementation chooses the smallest integer `q_C` satisfying

\[
P_C(q_C)\ge p_{\mathrm{target}}.
\]

Given classical setup cost `S_C`, candidate preparation cost `c_p`, verifier
evaluation cost `c_v`, readout cost `c_r`, and `M` reconstruction instances,

\[
C_C(M)=S_C+M\left[q_C(c_p+c_v)+c_r\right].
\]

## 4. Quantum matched-success plan

For every standard-Grover iteration count `r` up to the first peak, the ideal
one-run success is

\[
p_r=\sin^2\left((2r+1)\arcsin\sqrt{K/N}\right).
\]

Independent repetitions give success

\[
P_Q(r,s)=1-(1-p_r)^s.
\]

The smallest finite repetition count reaching the target is

\[
s_r=
\left\lceil
\frac{\log(1-p_{\mathrm{target}})}{\log(1-p_r)}
\right\rceil
\]

when `0 < p_r < p_target < 1`, with the exact boundary cases handled separately.
The optimizer evaluates every admissible `r` and chooses the pair `(r,s_r)` with
minimum modeled variable cost rather than assuming that the largest one-run
success is cheapest.

## 5. Logical resource pricing

For a selected plan, the compiler supplies exact or conservative logical counts:

- X, CNOT, H, and Z gates;
- decomposed T-count;
- logical qubits;
- oracle calls and Grover iterations;
- state preparation and measurement events.

A `FaultTolerantGateCosts` record assigns non-negative prices in one common
abstract unit. To avoid double counting, Toffoli gates are priced through their
reported decomposed T-count; the Toffoli count remains an auditable structural
metric. An optional logical-qubit/T-depth term can price a simple space-time
proxy.

For one run with resource vector `R_r`, define

\[
c_Q(r)=c_{\mathrm{load}}+c_{\mathrm{gates}}(R_r)+c_{\mathrm{readout}}.
\]

With compilation/setup cost `S_Q`,

\[
C_Q(M)=S_Q+M s_r c_Q(r).
\]

The report claims a modeled advantage only if

\[
C_Q(M)<C_C(M)
\]

at the same target success.

## 6. T-cost and amortization thresholds

For a fixed selected plan, write the quantum total as

\[
C_Q(M)=B_Q(M)+M N_T c_T,
\]

where `B_Q` excludes T gates and `N_T` is the T-count per instance including
repetitions. The open T-cost threshold compatible with strict advantage is

\[
c_T<\frac{C_C(M)-B_Q(M)}{MN_T}.
\]

A non-positive threshold proves that no non-negative T price can rescue the
selected plan under the other supplied assumptions.

For fixed per-instance costs

\[
V_C=q_C(c_p+c_v)+c_r,
\qquad
V_Q=s_rc_Q(r),
\]

compilation amortizes only when `V_Q < V_C`. The first workload with strict
advantage is

\[
M_{\min}
=
\left\lfloor
\frac{S_Q-S_C}{V_C-V_Q}
\right\rfloor+1,
\]

clamped to at least one and evaluated directly to preserve strictness.

## 7. Required reporting

Every cost table must include:

1. candidate definition, `N`, `K`, and target equivalence;
2. required and achieved success for both methods;
3. classical query count and per-query components;
4. selected Grover iterations and repetitions;
5. complete logical resource record from the actual compiled oracle;
6. compilation, loading, measurement, and readout costs;
7. common units and every numerical price;
8. T-cost threshold and compilation-amortization threshold when defined;
9. sensitivity sweeps showing both advantage and no-advantage regions;
10. a statement that the result is conditional unless tied to measured hardware
    or a fully specified fault-tolerant architecture.

## 8. Executable implementation

The implementation in `qrecon.oracles.costing` provides:

- `ClassicalSearchCosts`;
- `FaultTolerantGateCosts`;
- `QuantumSearchCosts`;
- `optimize_quantum_search_plan`;
- `compare_end_to_end_search_costs`;
- `maximum_t_cost_for_fixed_plan`;
- `minimum_instances_for_fixed_plan_advantage`.

`examples/affine_oracle_cost_report.py` emits a machine-readable report for a
uniquely marked signed affine threshold. It intentionally includes one scenario
with no modeled advantage and one low-cost sensitivity scenario. Neither is
presented as a hardware prediction.

## 9. Claim boundary

A favorable row in this cost model is necessary but not sufficient for a CCF-A
level end-to-end quantum-advantage claim. The paper must also justify Q-Access,
validate compiler correctness, certify precision, use a realistic candidate
prior, compare against the strongest task-matched classical search or inference
method, and show that favorable parameters are physically or operationally
plausible.
