from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

OverflowMode = Literal["raise", "saturate"]


def round_half_away_from_zero(value: float) -> int:
    """Round a finite real number to the nearest integer, breaking ties away from zero."""

    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("value must be finite")
    if converted >= 0.0:
        return math.floor(converted + 0.5)
    return math.ceil(converted - 0.5)


def round_shift_right(value: int, shift: int) -> int:
    """Divide an integer by ``2**shift`` with ties rounded away from zero."""

    if shift < 0:
        raise ValueError("shift must be non-negative")
    integer = int(value)
    if shift == 0:
        return integer
    magnitude = abs(integer)
    rounded = (magnitude + (1 << (shift - 1))) >> shift
    return rounded if integer >= 0 else -rounded


def rescale_code(value: int, source_fractional_bits: int, target_fractional_bits: int) -> int:
    """Rescale an unbounded fixed-point code without applying output overflow."""

    if source_fractional_bits < 0 or target_fractional_bits < 0:
        raise ValueError("fractional bit counts must be non-negative")
    delta = source_fractional_bits - target_fractional_bits
    if delta > 0:
        return round_shift_right(int(value), delta)
    if delta < 0:
        return int(value) << (-delta)
    return int(value)


@dataclass(frozen=True)
class FixedPointFormat:
    """Two's-complement or unsigned fixed-point word format.

    A stored integer code ``q`` represents ``q / 2**fractional_bits``. Arithmetic
    helpers deliberately separate exact unbounded accumulation from the final
    overflow policy so that range proofs can be audited before compilation.
    """

    bits: int
    fractional_bits: int = 0
    signed: bool = True
    overflow: OverflowMode = "raise"

    def __post_init__(self) -> None:
        if self.bits <= 0:
            raise ValueError("bits must be positive")
        if self.fractional_bits < 0:
            raise ValueError("fractional_bits must be non-negative")
        if self.overflow not in ("raise", "saturate"):
            raise ValueError("overflow must be 'raise' or 'saturate'")

    @property
    def modulus(self) -> int:
        return 1 << self.bits

    @property
    def mask(self) -> int:
        return self.modulus - 1

    @property
    def min_code(self) -> int:
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def max_code(self) -> int:
        return (1 << (self.bits - 1)) - 1 if self.signed else self.mask

    @property
    def scale(self) -> int:
        return 1 << self.fractional_bits

    def contains(self, code: int) -> bool:
        value = int(code)
        return self.min_code <= value <= self.max_code

    def require_code(self, code: int) -> int:
        value = int(code)
        if not self.contains(value):
            raise OverflowError(
                f"code {value} does not fit {self.bits}-bit "
                f"{'signed' if self.signed else 'unsigned'} format"
            )
        return value

    def apply_overflow(self, code: int) -> int:
        value = int(code)
        if self.contains(value):
            return value
        if self.overflow == "saturate":
            return min(self.max_code, max(self.min_code, value))
        return self.require_code(value)

    def quantize(self, value: float) -> int:
        return self.apply_overflow(round_half_away_from_zero(float(value) * self.scale))

    def dequantize(self, code: int) -> float:
        return self.require_code(code) / self.scale

    def code_to_word(self, code: int) -> int:
        return self.require_code(code) & self.mask

    def word_to_code(self, word: int) -> int:
        raw = int(word)
        if raw < 0 or raw > self.mask:
            raise ValueError(f"word must lie in [0, {self.mask}]")
        if self.signed and raw >= (1 << (self.bits - 1)):
            return raw - self.modulus
        return raw

    def code_to_bits(self, code: int) -> tuple[int, ...]:
        word = self.code_to_word(code)
        return tuple((word >> index) & 1 for index in range(self.bits))

    def bits_to_code(self, bits: tuple[int, ...] | list[int]) -> int:
        if len(bits) != self.bits:
            raise ValueError(f"expected {self.bits} bits")
        word = 0
        for index, bit in enumerate(bits):
            if bit not in (0, 1):
                raise ValueError("bits must be zero or one")
            word |= int(bit) << index
        return self.word_to_code(word)

    def requantize(self, code: int, source_fractional_bits: int) -> int:
        return self.apply_overflow(
            rescale_code(int(code), source_fractional_bits, self.fractional_bits)
        )
