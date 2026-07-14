from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

from .fixed_point import FixedPointFormat, rescale_code

Activation = Literal["identity", "relu"]
OutputMode = Literal["raw", "argmax", "binary_threshold"]


@dataclass(frozen=True)
class LayerRangeReport:
    raw_output_bounds: tuple[tuple[int, int], ...]
    encoded_output_bounds: tuple[tuple[int, int], ...]
    no_overflow: bool


@dataclass(frozen=True)
class NetworkRangeReport:
    layer_reports: tuple[LayerRangeReport, ...]
    no_overflow: bool


@dataclass(frozen=True)
class QuantizedAffineLayer:
    """Bit-exact affine layer followed by an optional ReLU and requantization."""

    weights: tuple[tuple[int, ...], ...]
    biases: tuple[int, ...]
    input_format: FixedPointFormat
    weight_format: FixedPointFormat
    bias_format: FixedPointFormat
    output_format: FixedPointFormat
    activation: Activation = "identity"

    def __post_init__(self) -> None:
        weights = tuple(tuple(int(value) for value in row) for row in self.weights)
        biases = tuple(int(value) for value in self.biases)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "biases", biases)

        if not weights or not weights[0]:
            raise ValueError("weights must contain at least one non-empty row")
        width = len(weights[0])
        if any(len(row) != width for row in weights):
            raise ValueError("all weight rows must have equal width")
        if len(biases) != len(weights):
            raise ValueError("one bias is required per output row")
        if self.activation not in ("identity", "relu"):
            raise ValueError("activation must be 'identity' or 'relu'")
        for row in weights:
            for value in row:
                self.weight_format.require_code(value)
        for value in biases:
            self.bias_format.require_code(value)

    @property
    def input_dimension(self) -> int:
        return len(self.weights[0])

    @property
    def output_dimension(self) -> int:
        return len(self.weights)

    @property
    def accumulator_fractional_bits(self) -> int:
        return self.input_format.fractional_bits + self.weight_format.fractional_bits

    def _aligned_bias(self, bias: int) -> int:
        return rescale_code(
            bias,
            self.bias_format.fractional_bits,
            self.accumulator_fractional_bits,
        )

    def evaluate_codes(self, inputs: Sequence[int]) -> tuple[int, ...]:
        values = tuple(self.input_format.require_code(value) for value in inputs)
        if len(values) != self.input_dimension:
            raise ValueError(f"expected {self.input_dimension} input codes")

        outputs: list[int] = []
        for row, bias in zip(self.weights, self.biases):
            accumulator = sum(value * weight for value, weight in zip(values, row))
            accumulator += self._aligned_bias(bias)
            output = rescale_code(
                accumulator,
                self.accumulator_fractional_bits,
                self.output_format.fractional_bits,
            )
            if self.activation == "relu":
                output = max(0, output)
            outputs.append(self.output_format.apply_overflow(output))
        return tuple(outputs)

    def range_report(
        self, input_bounds: Sequence[tuple[int, int]] | None = None
    ) -> LayerRangeReport:
        if input_bounds is None:
            bounds = tuple(
                (self.input_format.min_code, self.input_format.max_code)
                for _ in range(self.input_dimension)
            )
        else:
            bounds = tuple((int(low), int(high)) for low, high in input_bounds)
            if len(bounds) != self.input_dimension:
                raise ValueError(f"expected {self.input_dimension} input bounds")
            for low, high in bounds:
                if low > high:
                    raise ValueError("each lower bound must not exceed its upper bound")
                self.input_format.require_code(low)
                self.input_format.require_code(high)

        raw_bounds: list[tuple[int, int]] = []
        encoded_bounds: list[tuple[int, int]] = []
        safe = True
        for row, bias in zip(self.weights, self.biases):
            lower = self._aligned_bias(bias)
            upper = lower
            for weight, (input_low, input_high) in zip(row, bounds):
                if weight >= 0:
                    lower += input_low * weight
                    upper += input_high * weight
                else:
                    lower += input_high * weight
                    upper += input_low * weight
            lower = rescale_code(
                lower,
                self.accumulator_fractional_bits,
                self.output_format.fractional_bits,
            )
            upper = rescale_code(
                upper,
                self.accumulator_fractional_bits,
                self.output_format.fractional_bits,
            )
            if self.activation == "relu":
                lower = max(0, lower)
                upper = max(0, upper)
            raw_bounds.append((lower, upper))
            in_range = self.output_format.contains(lower) and self.output_format.contains(upper)
            safe = safe and in_range
            encoded_bounds.append(
                (
                    self.output_format.apply_overflow(lower),
                    self.output_format.apply_overflow(upper),
                )
                if self.output_format.overflow == "saturate" or in_range
                else (lower, upper)
            )
        return LayerRangeReport(tuple(raw_bounds), tuple(encoded_bounds), safe)


@dataclass(frozen=True)
class QuantizedNetwork:
    """A finite-word affine/ReLU network with a public bit-level evaluator."""

    layers: tuple[QuantizedAffineLayer, ...]
    output_mode: OutputMode = "raw"
    binary_threshold: int = 0

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        object.__setattr__(self, "layers", layers)
        if not layers:
            raise ValueError("at least one layer is required")
        if self.output_mode not in ("raw", "argmax", "binary_threshold"):
            raise ValueError("unsupported output_mode")
        for left, right in zip(layers, layers[1:]):
            if left.output_dimension != right.input_dimension:
                raise ValueError("adjacent layer dimensions do not match")
            if left.output_format != right.input_format:
                raise ValueError("adjacent fixed-point formats must match exactly")
        if self.output_mode == "binary_threshold" and layers[-1].output_dimension != 1:
            raise ValueError("binary_threshold requires exactly one final output")

    @property
    def input_format(self) -> FixedPointFormat:
        return self.layers[0].input_format

    @property
    def output_format(self) -> FixedPointFormat:
        return self.layers[-1].output_format

    @property
    def input_dimension(self) -> int:
        return self.layers[0].input_dimension

    @property
    def input_bits(self) -> int:
        return self.input_dimension * self.input_format.bits

    @property
    def output_dimension(self) -> int:
        return self.layers[-1].output_dimension

    @property
    def output_bits(self) -> int:
        if self.output_mode == "raw":
            return self.output_dimension * self.output_format.bits
        if self.output_mode == "binary_threshold":
            return 1
        return max(1, math.ceil(math.log2(self.output_dimension)))

    def evaluate_codes(self, inputs: Sequence[int]) -> tuple[int, ...]:
        values = tuple(int(value) for value in inputs)
        for layer in self.layers:
            values = layer.evaluate_codes(values)
        return values

    def decode_input_word(self, word: int) -> tuple[int, ...]:
        raw = int(word)
        if raw < 0 or raw >= (1 << self.input_bits):
            raise ValueError(f"input word must fit {self.input_bits} bits")
        mask = self.input_format.mask
        return tuple(
            self.input_format.word_to_code(
                (raw >> (index * self.input_format.bits)) & mask
            )
            for index in range(self.input_dimension)
        )

    def encode_input_codes(self, values: Sequence[int]) -> int:
        codes = tuple(self.input_format.require_code(value) for value in values)
        if len(codes) != self.input_dimension:
            raise ValueError(f"expected {self.input_dimension} input codes")
        word = 0
        for index, code in enumerate(codes):
            word |= self.input_format.code_to_word(code) << (
                index * self.input_format.bits
            )
        return word

    def evaluate_input_word(self, word: int) -> int:
        outputs = self.evaluate_codes(self.decode_input_word(word))
        if self.output_mode == "binary_threshold":
            return int(outputs[0] >= int(self.binary_threshold))
        if self.output_mode == "argmax":
            return max(range(len(outputs)), key=lambda index: (outputs[index], -index))

        output_word = 0
        for index, code in enumerate(outputs):
            output_word |= self.output_format.code_to_word(code) << (
                index * self.output_format.bits
            )
        return output_word

    def range_report(self) -> NetworkRangeReport:
        bounds: tuple[tuple[int, int], ...] = tuple(
            (self.input_format.min_code, self.input_format.max_code)
            for _ in range(self.input_dimension)
        )
        reports: list[LayerRangeReport] = []
        no_overflow = True
        for layer in self.layers:
            report = layer.range_report(bounds)
            reports.append(report)
            no_overflow = no_overflow and report.no_overflow
            bounds = report.encoded_output_bounds
        return NetworkRangeReport(tuple(reports), no_overflow)


def quantized_binary_logistic_regression(
    weights: Sequence[int],
    bias: int,
    *,
    input_format: FixedPointFormat,
    weight_format: FixedPointFormat,
    bias_format: FixedPointFormat,
    logit_format: FixedPointFormat,
    threshold: int = 0,
) -> QuantizedNetwork:
    """Construct a one-logit integer/fixed-point logistic decision model."""

    layer = QuantizedAffineLayer(
        weights=(tuple(int(value) for value in weights),),
        biases=(int(bias),),
        input_format=input_format,
        weight_format=weight_format,
        bias_format=bias_format,
        output_format=logit_format,
        activation="identity",
    )
    return QuantizedNetwork((layer,), output_mode="binary_threshold", binary_threshold=threshold)
