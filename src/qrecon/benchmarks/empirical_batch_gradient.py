from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from qrecon.oracles.compiler import TruthTableOracle
from qrecon.theory.empirical_batch_boundary import batch_two_recovery_dichotomy
from qrecon.theory.one_shot_loading import (
    amortized_explicit_table_probe_floor,
    one_shot_explicit_table_boundary,
)
from qrecon.theory.unknown_k import evaluate_bbht_schedule
from qrecon.theory.unknown_k_staged import certify_staged_bbht_uniform_success

from .candidate_loading import (
    EmpiricalCandidateLoadingReport,
    empirical_candidate_loading_report,
)
from .real_candidate_manifest import (
    CandidateLoader,
    CandidateQuantizationSpec,
    LoadedRealCandidateSet,
    RealBatchGradientManifest,
    load_real_candidate_set,
)
from .tensor_candidates import QuantizedCandidateAudit, audit_quantized_candidate_tensor


@dataclass(frozen=True)
class VectorTwoSumReport:
    target: tuple[int, ...]
    candidate_count: int
    observation_dimension: int
    contribution_buckets: int
    indexed_contributions: int
    complement_lookups: int
    solution_count: int
    solutions: tuple[tuple[int, int], ...]
    complete: bool = True

    @property
    def elementary_index_operations(self) -> int:
        return self.indexed_contributions + self.complement_lookups

    def to_dict(self) -> dict[str, object]:
        return {
            "target": list(self.target),
            "candidate_count": self.candidate_count,
            "observation_dimension": self.observation_dimension,
            "contribution_buckets": self.contribution_buckets,
            "indexed_contributions": self.indexed_contributions,
            "complement_lookups": self.complement_lookups,
            "elementary_index_operations": self.elementary_index_operations,
            "solution_count": self.solution_count,
            "solutions": [list(pair) for pair in self.solutions],
            "complete": self.complete,
            "expected_time_complexity": "O(N + Z)",
            "memory_complexity": "O(N)",
        }


@dataclass(frozen=True)
class GradientRangeCertificate:
    contribution_minimum: tuple[int, ...]
    contribution_maximum: tuple[int, ...]
    pair_sum_minimum: tuple[int, ...]
    pair_sum_maximum: tuple[int, ...]
    gradient_bits: int
    no_overflow: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EmpiricalBatchGradientPoint:
    quantization: CandidateQuantizationSpec
    feature_quantization: QuantizedCandidateAudit
    target_quantization: QuantizedCandidateAudit
    record_loading: EmpiricalCandidateLoadingReport
    model_weights: tuple[int, ...]
    model_bias: int
    gradient_range: GradientRangeCertificate
    candidate_count: int
    unordered_batch_count: int
    padded_pair_population: int
    unique_observation_count: int
    fibre_size_histogram: tuple[tuple[int, int], ...]
    global_exact_batch_bayes_success_uniform: float
    target_pair: tuple[int, int]
    target_pair_rank: int
    target_observation: tuple[int, ...]
    target_fibre_ranks: tuple[int, ...]
    target_fibre_pairs: tuple[tuple[int, int], ...]
    target_fibre_size: int
    target_conditional_exact_batch_success_uniform: float
    target_uniquely_identifiable: bool
    two_sum: VectorTwoSumReport
    two_sum_matches_exhaustive_fibre: bool
    predicate_reference_sha256: str
    predicate_reference_resources: dict[str, int | str]
    predicate_reference_basis_verified: bool | None
    predicate_reference_matches_fibre: bool
    predicate_construction: str
    bbht_certificate: dict[str, object] | None
    bbht_target_evaluation: dict[str, object] | None
    record_table_one_shot_boundary: dict[str, object] | None
    record_loading_amortization: tuple[dict[str, object], ...]
    pair_predicate_one_shot_boundary: dict[str, object] | None
    theory_boundary: dict[str, int | float | bool | str]
    asymptotic_verdict: str
    source_contract_satisfied: bool
    semantic_cross_checks_passed: bool

    @property
    def publication_gate_passed(self) -> bool:
        return self.source_contract_satisfied and self.semantic_cross_checks_passed

    def to_dict(self) -> dict[str, object]:
        return {
            "quantization": self.quantization.to_dict(),
            "feature_quantization": self.feature_quantization.to_dict(),
            "target_quantization": self.target_quantization.to_dict(),
            "record_loading": self.record_loading.to_dict(),
            "model_weights": list(self.model_weights),
            "model_bias": self.model_bias,
            "gradient_range": self.gradient_range.to_dict(),
            "candidate_count": self.candidate_count,
            "unordered_batch_count": self.unordered_batch_count,
            "padded_pair_population": self.padded_pair_population,
            "unique_observation_count": self.unique_observation_count,
            "fibre_size_histogram": [
                {"fibre_size": size, "number_of_fibres": count}
                for size, count in self.fibre_size_histogram
            ],
            "global_exact_batch_bayes_success_uniform": (
                self.global_exact_batch_bayes_success_uniform
            ),
            "target_pair": list(self.target_pair),
            "target_pair_rank": self.target_pair_rank,
            "target_observation": list(self.target_observation),
            "target_fibre_ranks": list(self.target_fibre_ranks),
            "target_fibre_pairs": [list(pair) for pair in self.target_fibre_pairs],
            "target_fibre_size": self.target_fibre_size,
            "target_conditional_exact_batch_success_uniform": (
                self.target_conditional_exact_batch_success_uniform
            ),
            "target_uniquely_identifiable": self.target_uniquely_identifiable,
            "two_sum": self.two_sum.to_dict(),
            "two_sum_matches_exhaustive_fibre": self.two_sum_matches_exhaustive_fibre,
            "predicate_reference_sha256": self.predicate_reference_sha256,
            "predicate_reference_resources": self.predicate_reference_resources,
            "predicate_reference_basis_verified": self.predicate_reference_basis_verified,
            "predicate_reference_matches_fibre": self.predicate_reference_matches_fibre,
            "predicate_construction": self.predicate_construction,
            "bbht_certificate": self.bbht_certificate,
            "bbht_target_evaluation": self.bbht_target_evaluation,
            "record_table_one_shot_boundary": self.record_table_one_shot_boundary,
            "record_loading_amortization": list(self.record_loading_amortization),
            "pair_predicate_one_shot_boundary": self.pair_predicate_one_shot_boundary,
            "theory_boundary": self.theory_boundary,
            "asymptotic_verdict": self.asymptotic_verdict,
            "source_contract_satisfied": self.source_contract_satisfied,
            "semantic_cross_checks_passed": self.semantic_cross_checks_passed,
            "publication_gate_passed": self.publication_gate_passed,
        }


@dataclass(frozen=True)
class EmpiricalBatchGradientFailure:
    quantization: CandidateQuantizationSpec
    error_type: str
    error_message: str
    error_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "quantization": self.quantization.to_dict(),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_sha256": self.error_sha256,
        }


@dataclass(frozen=True)
class RealBatchGradientPhaseDiagram:
    manifest: RealBatchGradientManifest
    loaded_candidates: LoadedRealCandidateSet
    points: tuple[EmpiricalBatchGradientPoint, ...]
    failures: tuple[EmpiricalBatchGradientFailure, ...]

    @property
    def complete(self) -> bool:
        return len(self.points) == len(self.manifest.quantizations) and not self.failures

    @property
    def publication_ready(self) -> bool:
        return (
            self.manifest.publication_mode
            and self.complete
            and all(point.publication_gate_passed for point in self.points)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": self.manifest.to_dict(),
            "manifest_sha256": self.manifest.sha256,
            "loaded_candidates": self.loaded_candidates.to_dict(),
            "points": [point.to_dict() for point in self.points],
            "failures": [failure.to_dict() for failure in self.failures],
            "complete": self.complete,
            "publication_ready": self.publication_ready,
            "central_dichotomy": {
                "nonunique_fibre": (
                    "If the target observation fibre has K>1 non-equivalent index pairs, "
                    "the original two-record batch has conditional Bayes success 1/K under "
                    "the uniform prior, independent of classical or quantum optimization."
                ),
                "unique_fibre": (
                    "If K=1, complete vector two-sum inversion costs O(N) expected time and "
                    "memory, while unstructured Grover search over choose(N,2) pairs uses "
                    "Theta(N) verifier calls. There is no query-exponent separation."
                ),
            },
            "claim_boundary": (
                "The truth-table predicate is an enumerative correctness reference. Its gate "
                "count is not a scalable quantum-advantage artifact; explicit data loading and "
                "the structure-aware two-sum baseline remain mandatory."
            ),
        }


def _vector_add(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    if len(left) != len(right):
        raise ValueError("vector dimensions differ")
    return tuple(int(a) + int(b) for a, b in zip(left, right))


def _vector_subtract(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    if len(left) != len(right):
        raise ValueError("vector dimensions differ")
    return tuple(int(a) - int(b) for a, b in zip(left, right))


def solve_vector_two_sum(
    contributions: Sequence[Sequence[int]], target: Sequence[int]
) -> VectorTwoSumReport:
    """Return every unordered pair whose contribution vectors sum to ``target``."""

    vectors = tuple(tuple(int(value) for value in row) for row in contributions)
    target_vector = tuple(int(value) for value in target)
    if len(vectors) < 2:
        raise ValueError("at least two contribution vectors are required")
    if not target_vector:
        raise ValueError("target must contain at least one component")
    if any(len(row) != len(target_vector) for row in vectors):
        raise ValueError("all contribution vectors must match the target dimension")

    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, vector in enumerate(vectors):
        buckets[vector].append(index)

    solutions: list[tuple[int, int]] = []
    for left_index, left in enumerate(vectors):
        complement = _vector_subtract(target_vector, left)
        for right_index in buckets.get(complement, ()):
            if left_index < right_index:
                solutions.append((left_index, right_index))

    return VectorTwoSumReport(
        target=target_vector,
        candidate_count=len(vectors),
        observation_dimension=len(target_vector),
        contribution_buckets=len(buckets),
        indexed_contributions=len(vectors),
        complement_lookups=len(vectors),
        solution_count=len(solutions),
        solutions=tuple(solutions),
    )


def _gradient_contributions(
    feature_codes: np.ndarray,
    target_codes: np.ndarray,
    weights: tuple[int, ...],
    bias: int,
) -> tuple[tuple[int, ...], ...]:
    features = np.asarray(feature_codes, dtype=np.int64)
    targets = np.asarray(target_codes, dtype=np.int64).reshape(features.shape[0], -1)
    if features.ndim != 2 or targets.shape[1] != 1:
        raise ValueError("gradient benchmark requires a feature matrix and one scalar target")
    if features.shape[1] != len(weights):
        raise ValueError("model weights do not match the selected feature dimension")

    result: list[tuple[int, ...]] = []
    for row, target_row in zip(features, targets):
        values = tuple(int(value) for value in row)
        target = int(target_row[0])
        residual = int(bias) + sum(weight * value for weight, value in zip(weights, values))
        residual -= target
        result.append(tuple(residual * value for value in values) + (residual,))
    return tuple(result)


def _unordered_pairs(candidate_count: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (left, right)
        for left in range(candidate_count)
        for right in range(left + 1, candidate_count)
    )


def _range_certificate(
    contributions: tuple[tuple[int, ...], ...],
    observations: tuple[tuple[int, ...], ...],
    gradient_bits: int,
) -> GradientRangeCertificate:
    dimension = len(contributions[0])
    contribution_min = tuple(min(row[index] for row in contributions) for index in range(dimension))
    contribution_max = tuple(max(row[index] for row in contributions) for index in range(dimension))
    pair_min = tuple(min(row[index] for row in observations) for index in range(dimension))
    pair_max = tuple(max(row[index] for row in observations) for index in range(dimension))
    lower = -(1 << (gradient_bits - 1))
    upper = (1 << (gradient_bits - 1)) - 1
    safe = all(
        lower <= value <= upper
        for bounds in (contribution_min, contribution_max, pair_min, pair_max)
        for value in bounds
    )
    return GradientRangeCertificate(
        contribution_minimum=contribution_min,
        contribution_maximum=contribution_max,
        pair_sum_minimum=pair_min,
        pair_sum_maximum=pair_max,
        gradient_bits=gradient_bits,
        no_overflow=safe,
    )


def _fibre_histogram(fibres: dict[tuple[int, ...], list[int]]) -> tuple[tuple[int, int], ...]:
    counts: dict[int, int] = defaultdict(int)
    for ranks in fibres.values():
        counts[len(ranks)] += 1
    return tuple(sorted(counts.items()))


def _bbht_payload(
    manifest: RealBatchGradientManifest,
    predicate: TruthTableOracle,
    marked: int,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if manifest.access_contract == "classical_only":
        return None, None
    population = 1 << predicate.input_bits
    if population > manifest.max_exact_population:
        return None, None
    certificate = certify_staged_bbht_uniform_success(
        population,
        manifest.target_success,
        minimum_marked=1,
        growth_factor=manifest.bbht_growth_factor,
        attempts_per_stage=manifest.bbht_attempts_per_stage,
        max_stages=manifest.bbht_max_stages,
        max_exact_population=manifest.max_exact_population,
    )
    evaluation = evaluate_bbht_schedule(certificate.schedule, marked)
    certificate_payload: dict[str, object] = {
        "population": population,
        "minimum_marked": certificate.minimum_marked,
        "target_success": certificate.target_success,
        "growth_factor": manifest.bbht_growth_factor,
        "attempts_per_stage": manifest.bbht_attempts_per_stage,
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
    evaluation_payload: dict[str, object] = {
        "marked": marked,
        "achieved_success": evaluation.achieved_success,
        "expected_phase_oracle_calls": evaluation.expected_phase_oracle_calls,
        "expected_verification_queries": evaluation.expected_verification_queries,
        "expected_total_oracle_calls": evaluation.expected_total_oracle_calls,
    }
    return certificate_payload, evaluation_payload


def _run_point(
    manifest: RealBatchGradientManifest,
    loaded: LoadedRealCandidateSet,
    quantization: CandidateQuantizationSpec,
) -> EmpiricalBatchGradientPoint:
    features = audit_quantized_candidate_tensor(
        loaded.features,
        quantization.bits_per_value,
        quantization.fractional_bits,
        signed=quantization.signed,
        overflow=quantization.overflow,
        max_minterm_table_bits=manifest.max_minterm_table_bits,
    )
    targets = audit_quantized_candidate_tensor(
        loaded.targets,
        quantization.bits_per_value,
        quantization.fractional_bits,
        signed=quantization.signed,
        overflow=quantization.overflow,
        max_minterm_table_bits=manifest.max_minterm_table_bits,
    )
    combined_codes = np.ascontiguousarray(
        np.concatenate(
            [
                features.codes.reshape(manifest.candidate_count, -1),
                targets.codes.reshape(manifest.candidate_count, -1),
            ],
            axis=1,
        )
    )
    record_loading = empirical_candidate_loading_report(
        combined_codes,
        quantization.bits_per_value,
        signed=quantization.signed,
        max_minterm_table_bits=manifest.max_minterm_table_bits,
    )

    weights = manifest.resolved_model_weights()
    contributions = _gradient_contributions(
        features.codes.reshape(manifest.candidate_count, -1),
        targets.codes,
        weights,
        manifest.model_bias,
    )
    pairs = _unordered_pairs(manifest.candidate_count)
    if len(pairs) > manifest.max_exact_batches:
        raise ValueError(
            f"unordered batch population {len(pairs)} exceeds max_exact_batches="
            f"{manifest.max_exact_batches}"
        )
    observations = tuple(
        _vector_add(contributions[left], contributions[right]) for left, right in pairs
    )
    gradient_range = _range_certificate(contributions, observations, manifest.gradient_bits)
    if not gradient_range.no_overflow:
        raise OverflowError("gradient_bits do not contain every contribution and pair sum")

    fibres: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for rank, observation in enumerate(observations):
        fibres[observation].append(rank)
    rank_by_pair = {pair: rank for rank, pair in enumerate(pairs)}
    target_pair = manifest.target_batch_indices
    target_rank = rank_by_pair[target_pair]
    target_observation = observations[target_rank]
    target_ranks = tuple(fibres[target_observation])
    target_pairs = tuple(pairs[rank] for rank in target_ranks)
    target_size = len(target_ranks)

    two_sum = solve_vector_two_sum(contributions, target_observation)
    exhaustive_match = tuple(sorted(two_sum.solutions)) == tuple(sorted(target_pairs))

    pair_count = len(pairs)
    input_bits = max(1, (pair_count - 1).bit_length())
    padded_population = 1 << input_bits
    marked_set = set(target_ranks)
    predicate = TruthTableOracle(
        input_bits=input_bits,
        output_bits=1,
        table=tuple(int(rank in marked_set) for rank in range(padded_population)),
        name="empirical_batch_gradient_pair_reference_predicate",
    )
    predicate_match = predicate.marked_inputs() == target_ranks
    basis_verified = (
        predicate.verify_basis_permutation()
        if predicate.input_bits <= manifest.max_basis_verification_bits
        else None
    )
    resources = predicate.resource_estimate(phase_kickback=True).to_dict()
    certificate_payload, evaluation_payload = _bbht_payload(
        manifest, predicate, target_size
    )

    if manifest.access_contract == "explicit_compilation":
        record_one_shot = one_shot_explicit_table_boundary(
            manifest.candidate_count,
            record_loading.word_bits,
            marked=1,
        ).to_dict()
        amortization = tuple(
            amortized_explicit_table_probe_floor(
                manifest.candidate_count,
                record_loading.word_bits,
                instances,
            ).to_dict()
            for instances in manifest.reusable_instances
        )
        predicate_one_shot = one_shot_explicit_table_boundary(
            pair_count,
            1,
            marked=target_size,
        ).to_dict()
    else:
        record_one_shot = None
        amortization = ()
        predicate_one_shot = None

    semantic_checks = (
        exhaustive_match
        and predicate_match
        and (basis_verified is not False)
        and two_sum.complete
    )
    source_contract = loaded.source_hash_matches is not False
    boundary = batch_two_recovery_dichotomy(
        manifest.candidate_count,
        target_size,
    )
    if target_size > 1:
        verdict = (
            "information_limited_exact_batch: the released aggregate gradient has a "
            f"{target_size}-element target fibre, so uniform conditional exact-index "
            f"success is at most 1/{target_size}; search quality cannot identify the "
            "original pair."
        )
    else:
        verdict = (
            "no_query_exponent_separation_for_batch_two: complete vector two-sum uses "
            "O(N) expected time and O(N) memory, while Grover over choose(N,2) pair "
            "indices uses Theta(N) verifier calls for the unique marked pair; loading and "
            "fault-tolerant oracle costs can only weaken the quantum side."
        )

    return EmpiricalBatchGradientPoint(
        quantization=quantization,
        feature_quantization=features,
        target_quantization=targets,
        record_loading=record_loading,
        model_weights=weights,
        model_bias=manifest.model_bias,
        gradient_range=gradient_range,
        candidate_count=manifest.candidate_count,
        unordered_batch_count=pair_count,
        padded_pair_population=padded_population,
        unique_observation_count=len(fibres),
        fibre_size_histogram=_fibre_histogram(fibres),
        global_exact_batch_bayes_success_uniform=len(fibres) / pair_count,
        target_pair=target_pair,
        target_pair_rank=target_rank,
        target_observation=target_observation,
        target_fibre_ranks=target_ranks,
        target_fibre_pairs=target_pairs,
        target_fibre_size=target_size,
        target_conditional_exact_batch_success_uniform=1.0 / target_size,
        target_uniquely_identifiable=target_size == 1,
        two_sum=two_sum,
        two_sum_matches_exhaustive_fibre=exhaustive_match,
        predicate_reference_sha256=predicate.truth_table_sha256,
        predicate_reference_resources=resources,
        predicate_reference_basis_verified=basis_verified,
        predicate_reference_matches_fibre=predicate_match,
        predicate_construction=(
            "enumerative truth-table correctness reference; building it consumes the exact "
            "pair fibre and is circular as a scalable search implementation"
        ),
        bbht_certificate=certificate_payload,
        bbht_target_evaluation=evaluation_payload,
        record_table_one_shot_boundary=record_one_shot,
        record_loading_amortization=amortization,
        pair_predicate_one_shot_boundary=predicate_one_shot,
        theory_boundary=boundary.to_dict(),
        asymptotic_verdict=verdict,
        source_contract_satisfied=source_contract,
        semantic_cross_checks_passed=semantic_checks,
    )


def run_real_batch_gradient_phase_diagram(
    manifest: RealBatchGradientManifest,
    *,
    loader_overrides: Mapping[str, CandidateLoader] | None = None,
) -> RealBatchGradientPhaseDiagram:
    """Run every declared precision while preserving all failed configurations."""

    loaded = load_real_candidate_set(
        manifest,
        loader_overrides=loader_overrides,
    )
    points: list[EmpiricalBatchGradientPoint] = []
    failures: list[EmpiricalBatchGradientFailure] = []
    for quantization in manifest.quantizations:
        try:
            points.append(_run_point(manifest, loaded, quantization))
        except Exception as exc:
            error_type = type(exc).__name__
            message = str(exc)
            digest = hashlib.sha256(f"{error_type}\n{message}".encode("utf-8")).hexdigest()
            failures.append(
                EmpiricalBatchGradientFailure(
                    quantization=quantization,
                    error_type=error_type,
                    error_message=message,
                    error_sha256=digest,
                )
            )
    return RealBatchGradientPhaseDiagram(
        manifest=manifest,
        loaded_candidates=loaded,
        points=tuple(points),
        failures=tuple(failures),
    )
