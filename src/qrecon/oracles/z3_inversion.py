from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import reduce
from operator import mul
from typing import Any, Literal, Sequence

from .fixed_point import FixedPointFormat, rescale_code
from .models import QuantizedAffineLayer

Termination = Literal["exhausted", "solution_limit", "unknown"]


@dataclass(frozen=True)
class Z3FixedPointInversionReport:
    input_dimension: int
    candidate_count: int
    target_codes: tuple[int, ...]
    solutions: tuple[tuple[int, ...], ...]
    solver_checks: int
    encoded_constraint_count: int
    termination: Termination
    reason_unknown: str | None
    timeout_ms: int | None

    @property
    def solution_count(self) -> int:
        return len(self.solutions)

    @property
    def complete(self) -> bool:
        return self.termination == "exhausted"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["target_codes"] = list(self.target_codes)
        payload["solutions"] = [list(solution) for solution in self.solutions]
        payload["solution_count"] = self.solution_count
        payload["complete"] = self.complete
        return payload


def _load_z3() -> Any:
    try:
        import z3  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Z3 inversion requires `pip install -e '.[solver]'`"
        ) from exc
    return z3


def _validate_two_layer_model(
    hidden_layer: QuantizedAffineLayer,
    output_layer: QuantizedAffineLayer,
) -> None:
    if hidden_layer.activation != "relu":
        raise ValueError("hidden_layer must use ReLU")
    if output_layer.activation != "identity":
        raise ValueError("output_layer must use identity activation")
    if hidden_layer.output_dimension != output_layer.input_dimension:
        raise ValueError("hidden and output dimensions do not match")
    if hidden_layer.output_format != output_layer.input_format:
        raise ValueError("hidden output and final input formats must match exactly")


def _validated_domains(
    layer: QuantizedAffineLayer,
    domains: Sequence[Sequence[int]] | None,
) -> tuple[tuple[int, ...], ...]:
    if domains is None:
        full = tuple(
            range(layer.input_format.min_code, layer.input_format.max_code + 1)
        )
        return tuple(full for _ in range(layer.input_dimension))
    if len(domains) != layer.input_dimension:
        raise ValueError("one candidate domain is required per input feature")
    result: list[tuple[int, ...]] = []
    for index, domain in enumerate(domains):
        values = tuple(dict.fromkeys(int(value) for value in domain))
        if not values:
            raise ValueError(f"domain {index} must be non-empty")
        for value in values:
            layer.input_format.require_code(value)
        result.append(values)
    return tuple(result)


def _rescale_expression(
    z3: Any,
    expression: Any,
    source_fractional_bits: int,
    target_fractional_bits: int,
) -> Any:
    delta = source_fractional_bits - target_fractional_bits
    if delta > 0:
        magnitude = z3.If(expression >= 0, expression, -expression)
        rounded_magnitude = (magnitude + (1 << (delta - 1))) / (1 << delta)
        return z3.If(expression >= 0, rounded_magnitude, -rounded_magnitude)
    if delta < 0:
        return expression * (1 << (-delta))
    return expression


def _overflow_expression(
    z3: Any,
    solver: Any,
    expression: Any,
    output_format: FixedPointFormat,
) -> Any:
    minimum = output_format.min_code
    maximum = output_format.max_code
    if output_format.overflow == "saturate":
        return z3.If(
            expression < minimum,
            minimum,
            z3.If(expression > maximum, maximum, expression),
        )
    solver.add(expression >= minimum, expression <= maximum)
    return expression


def _layer_expressions(
    z3: Any,
    solver: Any,
    layer: QuantizedAffineLayer,
    inputs: Sequence[Any],
) -> tuple[Any, ...]:
    if len(inputs) != layer.input_dimension:
        raise ValueError("symbolic input dimension does not match layer")
    outputs: list[Any] = []
    source_fractional_bits = layer.accumulator_fractional_bits
    for row, bias in zip(layer.weights, layer.biases):
        aligned_bias = rescale_code(
            bias,
            layer.bias_format.fractional_bits,
            source_fractional_bits,
        )
        accumulator = z3.IntVal(aligned_bias)
        for variable, weight in zip(inputs, row):
            accumulator = accumulator + variable * int(weight)
        rescaled = _rescale_expression(
            z3,
            accumulator,
            source_fractional_bits,
            layer.output_format.fractional_bits,
        )
        if layer.activation == "relu":
            rescaled = z3.If(rescaled < 0, 0, rescaled)
        outputs.append(
            _overflow_expression(z3, solver, rescaled, layer.output_format)
        )
    return tuple(outputs)


def solve_fixed_point_mlp_with_z3(
    hidden_layer: QuantizedAffineLayer,
    output_layer: QuantizedAffineLayer,
    target_codes: Sequence[int],
    *,
    domains: Sequence[Sequence[int]] | None = None,
    max_solutions: int | None = None,
    timeout_ms: int | None = None,
) -> Z3FixedPointInversionReport:
    """Exactly invert a two-layer fixed-point MLP with an SMT solver.

    The encoding uses unbounded integer arithmetic plus the same deterministic
    tie-away-from-zero rescaling, ReLU, range rejection, and saturation rules as
    `QuantizedAffineLayer.evaluate_codes`.  Candidate domains may be arbitrary
    finite code sets.  With no solution limit and a non-`unknown` solver result,
    returned solutions are complete.
    """

    z3 = _load_z3()
    _validate_two_layer_model(hidden_layer, output_layer)
    candidate_domains = _validated_domains(hidden_layer, domains)
    target = tuple(int(value) for value in target_codes)
    if len(target) != output_layer.output_dimension:
        raise ValueError("target dimension does not match final layer")
    for value in target:
        output_layer.output_format.require_code(value)
    if max_solutions is not None and max_solutions <= 0:
        raise ValueError("max_solutions must be positive or None")
    if timeout_ms is not None and timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive or None")

    solver = z3.Solver()
    if timeout_ms is not None:
        solver.set(timeout=int(timeout_ms))
    variables = tuple(
        z3.Int(f"qrecon_input_{index}")
        for index in range(hidden_layer.input_dimension)
    )
    for variable, domain in zip(variables, candidate_domains):
        solver.add(z3.Or(*(variable == value for value in domain)))

    hidden = _layer_expressions(z3, solver, hidden_layer, variables)
    outputs = _layer_expressions(z3, solver, output_layer, hidden)
    for expression, value in zip(outputs, target):
        solver.add(expression == value)

    encoded_constraint_count = len(solver.assertions())
    candidate_count = reduce(
        mul, (len(domain) for domain in candidate_domains), 1
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
            solver.add(z3.Or(*(variable != value for variable, value in zip(variables, solution))))
            continue
        if result == z3.unsat:
            termination = "exhausted"
            break
        termination = "unknown"
        reason_unknown = solver.reason_unknown()
        break

    return Z3FixedPointInversionReport(
        input_dimension=hidden_layer.input_dimension,
        candidate_count=candidate_count,
        target_codes=target,
        solutions=tuple(sorted(set(solutions))),
        solver_checks=solver_checks,
        encoded_constraint_count=encoded_constraint_count,
        termination=termination,
        reason_unknown=reason_unknown,
        timeout_ms=timeout_ms,
    )
