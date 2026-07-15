from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Sequence

from qrecon.oracles.domain_oracle import (
    ReversibleDomainRestrictedMLPEqualityOracle,
)
from qrecon.oracles.fixed_point import FixedPointFormat
from qrecon.oracles.fixed_point_inversion import (
    FixedPointInversionReport,
    solve_fixed_point_mlp_exact_output,
)
from qrecon.oracles.models import QuantizedAffineLayer
from qrecon.theory.unknown_k_staged import certify_staged_bbht_uniform_success


@dataclass(frozen=True)
class FixedPointMLPBenchmarkConfig:
    input_dimension: int = 2
    hidden_width: int = 2
    output_dimension: int = 2
    input_bits: int = 3
    fractional_bits: int = 1
    domain_codes: tuple[int, ...] = (-2, -1, 0, 1)
    target_success: float = 0.9
    max_exact_population: int = 4096
    max_basis_verification_bits: int = 8

    def __post_init__(self) -> None:
        if self.input_dimension <= 0 or self.hidden_width <= 0:
            raise ValueError("input_dimension and hidden_width must be positive")
        if self.output_dimension <= 0:
            raise ValueError("output_dimension must be positive")
        if self.input_bits <= 1:
            raise ValueError("input_bits must be at least two")
        if self.fractional_bits < 0:
            raise ValueError("fractional_bits must be non-negative")
        if not self.domain_codes:
            raise ValueError("domain_codes must be non-empty")
        if len(set(self.domain_codes)) != len(self.domain_codes):
            raise ValueError("domain_codes must not contain duplicates")
        if not 0.0 < self.target_success < 1.0:
            raise ValueError("target_success must lie strictly between zero and one")
        if self.max_exact_population <= 0 or self.max_basis_verification_bits <= 0:
            raise ValueError("benchmark limits must be positive")
        input_format = FixedPointFormat(
            self.input_bits, self.fractional_bits, True
        )
        for code in self.domain_codes:
            input_format.require_code(code)

    @property
    def candidate_count(self) -> int:
        return len(self.domain_codes) ** self.input_dimension

    @property
    def full_word_population(self) -> int:
        return 1 << (self.input_dimension * self.input_bits)


@dataclass(frozen=True)
class FixedPointMLPInstance:
    seed: int
    config: FixedPointMLPBenchmarkConfig
    hidden_layer: QuantizedAffineLayer
    output_layer: QuantizedAffineLayer
    domains: tuple[tuple[int, ...], ...]
    private_record: tuple[int, ...]
    target_codes: tuple[int, ...]


@dataclass(frozen=True)
class FixedPointMLPBenchmarkResult:
    seed: int
    config: FixedPointMLPBenchmarkConfig
    private_record: tuple[int, ...]
    target_codes: tuple[int, ...]
    candidate_count: int
    full_word_population: int
    solution_count: int
    uniquely_identifiable_on_domain: bool
    branch_and_bound: FixedPointInversionReport
    branch_and_bound_seconds: float
    oracle_build_seconds: float
    oracle_resources: dict[str, int | str]
    oracle_basis_permutation_verified: bool | None
    classical_oracle_solution_sets_match: bool
    bbht_certificate: dict[str, object] | None
    z3_report: dict[str, object] | None
    z3_seconds: float | None
    branch_and_bound_z3_solution_sets_match: bool | None

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "config": asdict(self.config),
            "private_record": list(self.private_record),
            "target_codes": list(self.target_codes),
            "candidate_count": self.candidate_count,
            "full_word_population": self.full_word_population,
            "solution_count": self.solution_count,
            "uniquely_identifiable_on_domain": self.uniquely_identifiable_on_domain,
            "branch_and_bound": self.branch_and_bound.to_dict(),
            "branch_and_bound_seconds": self.branch_and_bound_seconds,
            "oracle_build_seconds": self.oracle_build_seconds,
            "oracle_resources": self.oracle_resources,
            "oracle_basis_permutation_verified": (
                self.oracle_basis_permutation_verified
            ),
            "classical_oracle_solution_sets_match": (
                self.classical_oracle_solution_sets_match
            ),
            "bbht_certificate": self.bbht_certificate,
            "z3_report": self.z3_report,
            "z3_seconds": self.z3_seconds,
            "branch_and_bound_z3_solution_sets_match": (
                self.branch_and_bound_z3_solution_sets_match
            ),
        }


def _random_row(
    rng: random.Random, width: int, choices: tuple[int, ...]
) -> tuple[int, ...]:
    row = tuple(rng.choice(choices) for _ in range(width))
    if any(row):
        return row
    replacement = list(row)
    replacement[rng.randrange(width)] = rng.choice(tuple(value for value in choices if value))
    return tuple(replacement)


def build_fixed_point_mlp_instance(
    config: FixedPointMLPBenchmarkConfig,
    seed: int,
) -> FixedPointMLPInstance:
    """Generate one deterministic overflow-safe benchmark instance."""

    rng = random.Random(int(seed))
    input_format = FixedPointFormat(
        config.input_bits, config.fractional_bits, True
    )
    weight_format = FixedPointFormat(3, config.fractional_bits, True)
    weight_choices = tuple(
        value
        for value in (-2, -1, 0, 1, 2)
        if weight_format.contains(value)
    )
    hidden_bits = max(
        6,
        config.input_bits
        + math.ceil(math.log2(config.input_dimension + 1))
        + 3,
    )
    hidden_format = FixedPointFormat(
        hidden_bits, config.fractional_bits, True
    )
    hidden_bias_format = FixedPointFormat(
        hidden_bits + 2, 2 * config.fractional_bits, True
    )
    hidden = QuantizedAffineLayer(
        weights=tuple(
            _random_row(rng, config.input_dimension, weight_choices)
            for _ in range(config.hidden_width)
        ),
        biases=tuple(rng.choice((-1, 0, 1)) for _ in range(config.hidden_width)),
        input_format=input_format,
        weight_format=weight_format,
        bias_format=hidden_bias_format,
        output_format=hidden_format,
        activation="relu",
    )

    output_bits = max(
        8,
        hidden_bits + math.ceil(math.log2(config.hidden_width + 1)) + 3,
    )
    output_format = FixedPointFormat(
        output_bits, config.fractional_bits, True
    )
    output_bias_format = FixedPointFormat(
        output_bits + 2, 2 * config.fractional_bits, True
    )
    output = QuantizedAffineLayer(
        weights=tuple(
            _random_row(rng, config.hidden_width, weight_choices)
            for _ in range(config.output_dimension)
        ),
        biases=tuple(rng.choice((-1, 0, 1)) for _ in range(config.output_dimension)),
        input_format=hidden_format,
        weight_format=weight_format,
        bias_format=output_bias_format,
        output_format=output_format,
        activation="identity",
    )

    domains = tuple(tuple(config.domain_codes) for _ in range(config.input_dimension))
    private_record = tuple(
        rng.choice(config.domain_codes) for _ in range(config.input_dimension)
    )
    target = output.evaluate_codes(hidden.evaluate_codes(private_record))
    return FixedPointMLPInstance(
        seed=int(seed),
        config=config,
        hidden_layer=hidden,
        output_layer=output,
        domains=domains,
        private_record=private_record,
        target_codes=target,
    )


def _bbht_payload(
    config: FixedPointMLPBenchmarkConfig,
    *,
    target_success: float | None = None,
    growth_factor: float = 8.0 / 7.0,
    attempts_per_stage: int = 1,
    max_stages: int | None = None,
) -> dict[str, object] | None:
    population = config.full_word_population
    if population > config.max_exact_population:
        return None
    target = config.target_success if target_success is None else float(target_success)
    stages = 256 if max_stages is None else int(max_stages)
    certificate = certify_staged_bbht_uniform_success(
        population,
        target,
        growth_factor=growth_factor,
        attempts_per_stage=attempts_per_stage,
        max_stages=stages,
        max_exact_population=config.max_exact_population,
    )
    return {
        "minimum_marked": certificate.minimum_marked,
        "target_success": certificate.target_success,
        "growth_factor": float(growth_factor),
        "attempts_per_stage": int(attempts_per_stage),
        "maximum_stages": stages,
        "windows": list(certificate.schedule.windows),
        "rounds": certificate.schedule.rounds,
        "certified_minimum_success": certificate.certified_minimum_success,
        "worst_success_marked": certificate.worst_success_marked,
        "maximum_expected_phase_oracle_calls": (
            certificate.maximum_expected_phase_oracle_calls
        ),
        "maximum_expected_verification_queries": (
            certificate.maximum_expected_verification_queries
        ),
        "worst_case_total_oracle_calls": (
            certificate.schedule.worst_case_total_oracle_calls
        ),
    }


def run_fixed_point_mlp_benchmark(
    config: FixedPointMLPBenchmarkConfig,
    seed: int,
    *,
    use_z3: bool = False,
    z3_timeout_ms: int | None = None,
    target_success: float | None = None,
    bbht_growth_factor: float = 8.0 / 7.0,
    bbht_attempts_per_stage: int = 1,
    bbht_max_stages: int | None = None,
) -> FixedPointMLPBenchmarkResult:
    """Run matched BnB, optional SMT, domain oracle, and search audit."""

    instance = build_fixed_point_mlp_instance(config, seed)

    start = time.perf_counter()
    branch = solve_fixed_point_mlp_exact_output(
        instance.hidden_layer,
        instance.output_layer,
        instance.target_codes,
        domains=instance.domains,
    )
    branch_seconds = time.perf_counter() - start

    start = time.perf_counter()
    oracle = ReversibleDomainRestrictedMLPEqualityOracle(
        instance.hidden_layer,
        instance.output_layer,
        instance.target_codes,
        instance.domains,
        max_enumeration_bits=max(
            config.max_basis_verification_bits,
            instance.hidden_layer.input_dimension
            * instance.hidden_layer.input_format.bits,
        ),
    )
    oracle_seconds = time.perf_counter() - start
    branch_words = tuple(
        sorted(oracle.encode_inputs(solution) for solution in branch.solutions)
    )
    oracle_words = oracle.marked_inputs()
    basis_verified = (
        oracle.verify_basis_permutation()
        if oracle.input_bits <= config.max_basis_verification_bits
        else None
    )

    z3_report: dict[str, object] | None = None
    z3_seconds: float | None = None
    z3_match: bool | None = None
    if use_z3:
        from qrecon.oracles.z3_inversion import solve_fixed_point_mlp_with_z3

        start = time.perf_counter()
        z3_result = solve_fixed_point_mlp_with_z3(
            instance.hidden_layer,
            instance.output_layer,
            instance.target_codes,
            domains=instance.domains,
            timeout_ms=z3_timeout_ms,
        )
        z3_seconds = time.perf_counter() - start
        z3_report = z3_result.to_dict()
        z3_match = (
            z3_result.complete
            and tuple(sorted(z3_result.solutions))
            == tuple(sorted(branch.solutions))
        )

    return FixedPointMLPBenchmarkResult(
        seed=int(seed),
        config=config,
        private_record=instance.private_record,
        target_codes=instance.target_codes,
        candidate_count=config.candidate_count,
        full_word_population=config.full_word_population,
        solution_count=branch.solution_count,
        uniquely_identifiable_on_domain=branch.solution_count == 1,
        branch_and_bound=branch,
        branch_and_bound_seconds=branch_seconds,
        oracle_build_seconds=oracle_seconds,
        oracle_resources=oracle.resource_estimate(phase_kickback=True).to_dict(),
        oracle_basis_permutation_verified=basis_verified,
        classical_oracle_solution_sets_match=branch_words == oracle_words,
        bbht_certificate=_bbht_payload(
            config,
            target_success=target_success,
            growth_factor=bbht_growth_factor,
            attempts_per_stage=bbht_attempts_per_stage,
            max_stages=bbht_max_stages,
        ),
        z3_report=z3_report,
        z3_seconds=z3_seconds,
        branch_and_bound_z3_solution_sets_match=z3_match,
    )


def run_fixed_point_mlp_benchmark_matrix(
    configs: Sequence[FixedPointMLPBenchmarkConfig],
    seeds: Sequence[int],
    *,
    use_z3: bool = False,
    z3_timeout_ms: int | None = None,
) -> tuple[FixedPointMLPBenchmarkResult, ...]:
    if not configs or not seeds:
        raise ValueError("configs and seeds must both be non-empty")
    return tuple(
        run_fixed_point_mlp_benchmark(
            config,
            seed,
            use_z3=use_z3,
            z3_timeout_ms=z3_timeout_ms,
        )
        for config in configs
        for seed in seeds
    )
