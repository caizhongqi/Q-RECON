from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import reduce
from operator import mul
from typing import Sequence

from .fixed_point import rescale_code
from .models import QuantizedAffineLayer


@dataclass(frozen=True)
class CodeInterval:
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError("interval minimum cannot exceed maximum")

    def contains(self, value: int) -> bool:
        return self.minimum <= int(value) <= self.maximum


@dataclass(frozen=True)
class FixedPointInversionReport:
    input_dimension: int
    candidate_count: int
    target_codes: tuple[int, ...]
    variable_order: tuple[int, ...]
    nodes_visited: int
    bound_evaluations: int
    pruned_nodes: int
    leaf_evaluations: int
    solutions: tuple[tuple[int, ...], ...]
    stopped_early: bool

    @property
    def solution_count(self) -> int:
        return len(self.solutions)

    @property
    def leaf_fraction(self) -> float:
        if self.candidate_count == 0:
            return 0.0
        return self.leaf_evaluations / self.candidate_count

    @property
    def exhaustive_leaf_reduction(self) -> float:
        if self.candidate_count == 0:
            return 0.0
        return 1.0 - self.leaf_fraction

    def to_dict(self) -> dict[str, int | float | bool | list[int] | list[list[int]]]:
        payload = asdict(self)
        payload["target_codes"] = list(self.target_codes)
        payload["variable_order"] = list(self.variable_order)
        payload["solutions"] = [list(solution) for solution in self.solutions]
        payload["solution_count"] = self.solution_count
        payload["leaf_fraction"] = self.leaf_fraction
        payload["exhaustive_leaf_reduction"] = self.exhaustive_leaf_reduction
        return payload


def _validate_two_layer_fixed_point_mlp(
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
        full = tuple(range(layer.input_format.min_code, layer.input_format.max_code + 1))
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


def _aligned_bias(layer: QuantizedAffineLayer, bias: int) -> tuple[int, int]:
    product_fractional_bits = (
        layer.input_format.fractional_bits + layer.weight_format.fractional_bits
    )
    return (
        rescale_code(
            bias,
            layer.bias_format.fractional_bits,
            product_fractional_bits,
        ),
        product_fractional_bits,
    )


def _affine_output_intervals(
    layer: QuantizedAffineLayer,
    input_intervals: Sequence[CodeInterval],
) -> tuple[CodeInterval, ...]:
    if len(input_intervals) != layer.input_dimension:
        raise ValueError("input interval dimension does not match layer")
    output: list[CodeInterval] = []
    for row, bias in zip(layer.weights, layer.biases):
        aligned_bias, product_fractional_bits = _aligned_bias(layer, bias)
        lower = aligned_bias
        upper = aligned_bias
        for weight, interval in zip(row, input_intervals):
            if weight >= 0:
                lower += weight * interval.minimum
                upper += weight * interval.maximum
            else:
                lower += weight * interval.maximum
                upper += weight * interval.minimum
        quantized_lower = layer.output_format.requantize(
            lower, product_fractional_bits
        )
        quantized_upper = layer.output_format.requantize(
            upper, product_fractional_bits
        )
        if quantized_lower > quantized_upper:
            raise ArithmeticError("monotone requantization interval was inverted")
        if layer.activation == "relu":
            quantized_lower = max(0, quantized_lower)
            quantized_upper = max(0, quantized_upper)
        output.append(CodeInterval(quantized_lower, quantized_upper))
    return tuple(output)


def fixed_point_mlp_output_bounds(
    hidden_layer: QuantizedAffineLayer,
    output_layer: QuantizedAffineLayer,
    domains: Sequence[Sequence[int]] | None = None,
    partial_assignment: Sequence[int | None] | None = None,
) -> tuple[CodeInterval, ...]:
    """Return a sound interval enclosure for every final output code.

    Bounds use interval propagation through exact monotone requantization and
    ReLU. They may be loose because correlations between hidden units are
    discarded, but they never exclude a realizable output under the declared
    fixed-point semantics.
    """

    _validate_two_layer_fixed_point_mlp(hidden_layer, output_layer)
    candidate_domains = _validated_domains(hidden_layer, domains)
    if partial_assignment is None:
        assignment = (None,) * hidden_layer.input_dimension
    else:
        if len(partial_assignment) != hidden_layer.input_dimension:
            raise ValueError("partial assignment dimension does not match input")
        assignment = tuple(partial_assignment)

    input_intervals: list[CodeInterval] = []
    for index, (value, domain) in enumerate(zip(assignment, candidate_domains)):
        if value is None:
            input_intervals.append(CodeInterval(min(domain), max(domain)))
        else:
            code = int(value)
            if code not in domain:
                raise ValueError(f"assigned code for feature {index} is outside its domain")
            input_intervals.append(CodeInterval(code, code))

    hidden_intervals = _affine_output_intervals(hidden_layer, input_intervals)
    return _affine_output_intervals(output_layer, hidden_intervals)


def _variable_order(
    hidden_layer: QuantizedAffineLayer,
    domains: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    scores: list[tuple[int, int, int]] = []
    for feature in range(hidden_layer.input_dimension):
        influence = sum(abs(row[feature]) for row in hidden_layer.weights)
        span = max(domains[feature]) - min(domains[feature])
        scores.append((influence * max(1, span), len(domains[feature]), feature))
    return tuple(
        feature
        for _, _, feature in sorted(
            scores,
            key=lambda item: (-item[0], -item[1], item[2]),
        )
    )


def solve_fixed_point_mlp_exact_output(
    hidden_layer: QuantizedAffineLayer,
    output_layer: QuantizedAffineLayer,
    target_codes: Sequence[int],
    *,
    domains: Sequence[Sequence[int]] | None = None,
    max_solutions: int | None = None,
) -> FixedPointInversionReport:
    """Exactly invert a two-layer fixed-point MLP with sound branch-and-bound.

    The solver is complete when ``max_solutions`` is ``None``. With a positive
    limit it returns the first solutions under a deterministic influence-based
    variable order and records that search stopped early.
    """

    _validate_two_layer_fixed_point_mlp(hidden_layer, output_layer)
    candidates = _validated_domains(hidden_layer, domains)
    target = tuple(int(value) for value in target_codes)
    if len(target) != output_layer.output_dimension:
        raise ValueError("target dimension does not match final layer")
    for value in target:
        output_layer.output_format.require_code(value)
    if max_solutions is not None and max_solutions <= 0:
        raise ValueError("max_solutions must be positive or None")

    order = _variable_order(hidden_layer, candidates)
    candidate_count = reduce(mul, (len(domain) for domain in candidates), 1)
    assignment: list[int | None] = [None] * hidden_layer.input_dimension
    solutions: list[tuple[int, ...]] = []
    nodes_visited = 0
    bound_evaluations = 0
    pruned_nodes = 0
    leaf_evaluations = 0
    stopped_early = False

    def target_possible(bounds: tuple[CodeInterval, ...]) -> bool:
        return all(interval.contains(value) for interval, value in zip(bounds, target))

    def search(depth: int) -> bool:
        nonlocal nodes_visited, bound_evaluations, pruned_nodes
        nonlocal leaf_evaluations, stopped_early
        nodes_visited += 1
        bounds = fixed_point_mlp_output_bounds(
            hidden_layer,
            output_layer,
            candidates,
            assignment,
        )
        bound_evaluations += 1
        if not target_possible(bounds):
            pruned_nodes += 1
            return False

        if depth == len(order):
            leaf_evaluations += 1
            record = tuple(int(value) for value in assignment if value is not None)
            hidden = hidden_layer.evaluate_codes(record)
            observed = output_layer.evaluate_codes(hidden)
            if observed == target:
                solutions.append(record)
                if max_solutions is not None and len(solutions) >= max_solutions:
                    stopped_early = True
                    return True
            return False

        feature = order[depth]
        for value in candidates[feature]:
            assignment[feature] = value
            if search(depth + 1):
                assignment[feature] = None
                return True
        assignment[feature] = None
        return False

    search(0)
    return FixedPointInversionReport(
        input_dimension=hidden_layer.input_dimension,
        candidate_count=candidate_count,
        target_codes=target,
        variable_order=order,
        nodes_visited=nodes_visited,
        bound_evaluations=bound_evaluations,
        pruned_nodes=pruned_nodes,
        leaf_evaluations=leaf_evaluations,
        solutions=tuple(solutions),
        stopped_early=stopped_early,
    )


def exhaustive_fixed_point_mlp_solutions(
    hidden_layer: QuantizedAffineLayer,
    output_layer: QuantizedAffineLayer,
    target_codes: Sequence[int],
    *,
    domains: Sequence[Sequence[int]] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Reference exhaustive solver used to validate branch-and-bound results."""

    _validate_two_layer_fixed_point_mlp(hidden_layer, output_layer)
    candidates = _validated_domains(hidden_layer, domains)
    target = tuple(int(value) for value in target_codes)
    if len(target) != output_layer.output_dimension:
        raise ValueError("target dimension does not match final layer")
    for value in target:
        output_layer.output_format.require_code(value)

    result: list[tuple[int, ...]] = []

    def enumerate_assignments(index: int, prefix: list[int]) -> None:
        if index == len(candidates):
            record = tuple(prefix)
            hidden = hidden_layer.evaluate_codes(record)
            if output_layer.evaluate_codes(hidden) == target:
                result.append(record)
            return
        for value in candidates[index]:
            prefix.append(value)
            enumerate_assignments(index + 1, prefix)
            prefix.pop()

    enumerate_assignments(0, [])
    return tuple(result)
