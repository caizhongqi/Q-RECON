from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .data_loading import explicit_table_compiler_bit_probe_lower_bound


def _finite_non_negative(name: str, value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted


@dataclass(frozen=True)
class PositiveIntegerWorkloadRegion:
    """Contiguous positive-integer workload region.

    ``maximum=None`` denotes an unbounded upper end. The empty set is represented
    by ``None`` at API boundaries rather than by an invalid instance of this class.
    """

    minimum: int
    maximum: int | None = None

    def __post_init__(self) -> None:
        if self.minimum <= 0:
            raise ValueError("minimum must be positive")
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("maximum must be at least minimum")

    def contains(self, instances: int) -> bool:
        value = int(instances)
        return value >= self.minimum and (
            self.maximum is None or value <= self.maximum
        )

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)


@dataclass(frozen=True)
class ExplicitTableNoAdvantageCertificate:
    """One common-unit lower/upper-bound comparison for an explicit table.

    The quantum lower bound includes the worst-case exact-compiler requirement to
    inspect all ``entries * word_bits`` table-description bits. The classical
    quantity is a caller-supplied *upper* bound for a complete matched pipeline.
    Whenever the quantum lower bound is at least the classical upper bound, strict
    quantum cost advantage is impossible under the declared common unit.
    """

    entries: int
    word_bits: int
    instances: int
    table_description_bits: int
    quantum_setup_lower_bound: float
    quantum_variable_lower_bound: float
    classical_setup_upper_bound: float
    classical_variable_upper_bound: float
    quantum_total_lower_bound: float
    classical_total_upper_bound: float
    certified_no_advantage: bool

    @property
    def no_advantage_margin(self) -> float:
        return self.quantum_total_lower_bound - self.classical_total_upper_bound

    def to_dict(self) -> dict[str, int | float | bool]:
        result = asdict(self)
        result["no_advantage_margin"] = self.no_advantage_margin
        return result


def certify_explicit_table_no_advantage(
    entries: int,
    word_bits: int,
    *,
    instances: int = 1,
    quantum_setup_extra_lower_bound: float = 0.0,
    quantum_variable_lower_bound: float = 0.0,
    classical_setup_upper_bound: float = 0.0,
    classical_variable_upper_bound: float,
) -> ExplicitTableNoAdvantageCertificate:
    """Certify a strict no-advantage point in table-bit-probe-equivalent units.

    This helper assumes explicit compilation of an arbitrary empirical table. A
    physical QRAM already containing the table or a succinct generator is a
    different access model and must not use this compiler bit-probe lower bound.
    """

    count = int(entries)
    width = int(word_bits)
    workload = int(instances)
    if workload <= 0:
        raise ValueError("instances must be positive")
    description_bits = explicit_table_compiler_bit_probe_lower_bound(count, width)
    quantum_extra = _finite_non_negative(
        "quantum_setup_extra_lower_bound", quantum_setup_extra_lower_bound
    )
    quantum_variable = _finite_non_negative(
        "quantum_variable_lower_bound", quantum_variable_lower_bound
    )
    classical_setup = _finite_non_negative(
        "classical_setup_upper_bound", classical_setup_upper_bound
    )
    classical_variable = _finite_non_negative(
        "classical_variable_upper_bound", classical_variable_upper_bound
    )

    quantum_setup = float(description_bits) + quantum_extra
    quantum_total = quantum_setup + workload * quantum_variable
    classical_total = classical_setup + workload * classical_variable
    return ExplicitTableNoAdvantageCertificate(
        entries=count,
        word_bits=width,
        instances=workload,
        table_description_bits=description_bits,
        quantum_setup_lower_bound=quantum_setup,
        quantum_variable_lower_bound=quantum_variable,
        classical_setup_upper_bound=classical_setup,
        classical_variable_upper_bound=classical_variable,
        quantum_total_lower_bound=quantum_total,
        classical_total_upper_bound=classical_total,
        certified_no_advantage=quantum_total >= classical_total,
    )


def _linear_non_negative_region(
    intercept: float, slope: float
) -> PositiveIntegerWorkloadRegion | None:
    """Positive integers ``M`` satisfying ``intercept + slope*M >= 0``."""

    if slope == 0.0:
        return PositiveIntegerWorkloadRegion(1, None) if intercept >= 0.0 else None
    if slope > 0.0:
        minimum = max(1, math.ceil(-intercept / slope))
        while intercept + slope * minimum < 0.0:
            minimum += 1
        while minimum > 1 and intercept + slope * (minimum - 1) >= 0.0:
            minimum -= 1
        return PositiveIntegerWorkloadRegion(minimum, None)

    maximum = math.floor(intercept / (-slope))
    while maximum >= 1 and intercept + slope * maximum < 0.0:
        maximum -= 1
    while intercept + slope * (maximum + 1) >= 0.0:
        maximum += 1
    if maximum < 1:
        return None
    return PositiveIntegerWorkloadRegion(1, maximum)


def certified_explicit_table_no_advantage_region(
    entries: int,
    word_bits: int,
    *,
    quantum_setup_extra_lower_bound: float = 0.0,
    quantum_variable_lower_bound: float = 0.0,
    classical_setup_upper_bound: float = 0.0,
    classical_variable_upper_bound: float,
) -> PositiveIntegerWorkloadRegion | None:
    """All workloads where explicit loading alone rules out strict advantage.

    The returned region solves exactly

    ``N*w + Q_setup_extra + M*Q_variable_lower``
    ``>= C_setup_upper + M*C_variable_upper``.

    A bounded region means table loading rules out advantage only before enough
    reuse amortizes setup. An unbounded region means the declared quantum lower
    bound never drops below the complete classical upper bound.
    """

    description_bits = explicit_table_compiler_bit_probe_lower_bound(
        int(entries), int(word_bits)
    )
    quantum_extra = _finite_non_negative(
        "quantum_setup_extra_lower_bound", quantum_setup_extra_lower_bound
    )
    quantum_variable = _finite_non_negative(
        "quantum_variable_lower_bound", quantum_variable_lower_bound
    )
    classical_setup = _finite_non_negative(
        "classical_setup_upper_bound", classical_setup_upper_bound
    )
    classical_variable = _finite_non_negative(
        "classical_variable_upper_bound", classical_variable_upper_bound
    )
    intercept = description_bits + quantum_extra - classical_setup
    slope = quantum_variable - classical_variable
    return _linear_non_negative_region(float(intercept), float(slope))
