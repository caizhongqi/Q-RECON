from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Sequence

from .models import QuantizedAffineLayer, QuantizedNetwork
from .z3_inversion import (
    Termination,
    Z3FixedPointInversionReport,
    _layer_expressions,
    _load_z3,
    _validated_domains,
)


def solve_fixed_point_deep_mlp_with_z3(
    layers: Sequence[QuantizedAffineLayer],
    target_codes: Sequence[int],
    *,
    domains: Sequence[Sequence[int]] | None = None,
    max_solutions: int | None = None,
    timeout_ms: int | None = None,
) -> Z3FixedPointInversionReport:
    """Exactly invert an arbitrary-depth fixed-point affine/ReLU network.

    The network uses the same unbounded-integer accumulator, deterministic
    tie-away-from-zero rescaling, activation, range-rejection, and saturation
    semantics as :class:`QuantizedNetwork`. Candidate domains may be arbitrary
    finite code sets. If no solution limit is supplied and Z3 returns ``unsat``
    after model blocking, the reported solution fibre is complete.

    The final layer is required to use identity activation so the target is a
    public raw output code vector, matching the arbitrary-depth coherent value
    and equality-oracle contract.
    """

    declared = tuple(layers)
    network = QuantizedNetwork(declared, output_mode="raw")
    if declared[-1].activation != "identity":
        raise ValueError("the final fixed-point layer must use identity activation")

    candidate_domains = _validated_domains(declared[0], domains)
    target = tuple(int(value) for value in target_codes)
    if len(target) != network.output_dimension:
        raise ValueError("target dimension does not match final layer")
    for value in target:
        network.output_format.require_code(value)
    if max_solutions is not None and max_solutions <= 0:
        raise ValueError("max_solutions must be positive or None")
    if timeout_ms is not None and timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive or None")

    z3 = _load_z3()
    solver = z3.Solver()
    if timeout_ms is not None:
        solver.set(timeout=int(timeout_ms))

    variables = tuple(
        z3.Int(f"qrecon_deep_input_{index}")
        for index in range(network.input_dimension)
    )
    for variable, domain in zip(variables, candidate_domains):
        solver.add(z3.Or(*(variable == value for value in domain)))

    expressions: tuple[object, ...] = tuple(variables)
    for layer in declared:
        expressions = _layer_expressions(z3, solver, layer, expressions)
    for expression, value in zip(expressions, target):
        solver.add(expression == value)

    encoded_constraint_count = len(solver.assertions())
    candidate_count = reduce(
        mul,
        (len(domain) for domain in candidate_domains),
        1,
    )
    solutions: list[tuple[int, ...]] = []
    solver_checks = 0
    termination: Termination = "unknown"
    reason_unknown: str | None = None

    while True:
        result = solver.check()
        solver_checks += 1
        if result == z3.sat:
            model = solver.model()
            solution = tuple(
                int(model.eval(variable, model_completion=True).as_long())
                for variable in variables
            )
            solutions.append(solution)
            if max_solutions is not None and len(solutions) >= max_solutions:
                termination = "solution_limit"
                break
            solver.add(
                z3.Or(
                    *(
                        variable != value
                        for variable, value in zip(variables, solution)
                    )
                )
            )
            continue
        if result == z3.unsat:
            termination = "exhausted"
            break
        termination = "unknown"
        reason_unknown = solver.reason_unknown()
        break

    return Z3FixedPointInversionReport(
        input_dimension=network.input_dimension,
        candidate_count=candidate_count,
        target_codes=target,
        solutions=tuple(sorted(set(solutions))),
        solver_checks=solver_checks,
        encoded_constraint_count=encoded_constraint_count,
        termination=termination,
        reason_unknown=reason_unknown,
        timeout_ms=timeout_ms,
    )
