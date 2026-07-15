from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from qrecon.theory.data_loading import (
    ExplicitLookupDescription,
    LookupMintermResources,
    TypicalLookupCircuitLowerBound,
    explicit_lookup_description,
    explicit_table_compiler_bit_probe_lower_bound,
    lookup_minterm_resources,
    typical_lookup_circuit_lower_bound,
)


@dataclass(frozen=True)
class EmpiricalCandidateLoadingReport:
    candidate_count: int
    unique_candidate_count: int
    duplicate_candidate_count: int
    candidate_shape: tuple[int, ...]
    values_per_candidate: int
    bits_per_value: int
    signed: bool
    word_bits: int
    source_sha256: str
    exact_index_bayes_success_uniform: float
    explicit_lookup: ExplicitLookupDescription
    deduplicated_lookup: ExplicitLookupDescription
    compiler_bit_probe_lower_bound: int
    typical_circuit_lower_bound: TypicalLookupCircuitLowerBound
    minterm_resources: LookupMintermResources | None
    minterm_skipped_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "unique_candidate_count": self.unique_candidate_count,
            "duplicate_candidate_count": self.duplicate_candidate_count,
            "candidate_shape": list(self.candidate_shape),
            "values_per_candidate": self.values_per_candidate,
            "bits_per_value": self.bits_per_value,
            "signed": self.signed,
            "word_bits": self.word_bits,
            "source_sha256": self.source_sha256,
            "exact_index_bayes_success_uniform": self.exact_index_bayes_success_uniform,
            "explicit_lookup": self.explicit_lookup.to_dict(),
            "deduplicated_lookup": self.deduplicated_lookup.to_dict(),
            "compiler_bit_probe_lower_bound": self.compiler_bit_probe_lower_bound,
            "typical_circuit_lower_bound": self.typical_circuit_lower_bound.to_dict(),
            "minterm_resources": (
                None if self.minterm_resources is None else self.minterm_resources.to_dict()
            ),
            "minterm_skipped_reason": self.minterm_skipped_reason,
        }


def _validate_codes(
    candidates: np.ndarray, bits_per_value: int, signed: bool
) -> np.ndarray:
    bits = int(bits_per_value)
    if bits <= 0:
        raise ValueError("bits_per_value must be positive")
    array = np.asarray(candidates)
    if array.ndim < 2:
        raise ValueError("candidates must have shape (candidate, ...)")
    if array.shape[0] <= 0:
        raise ValueError("candidates must be non-empty")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError("candidate codes must be finite integers")
        rounded = np.rint(array)
        if not np.array_equal(array, rounded):
            raise ValueError("candidate codes must be integers")
        array = rounded.astype(np.int64)
    else:
        array = array.astype(np.int64, copy=False)

    minimum = -(1 << (bits - 1)) if signed else 0
    maximum = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    if int(array.min()) < minimum or int(array.max()) > maximum:
        raise ValueError(
            f"candidate codes must lie in [{minimum}, {maximum}] for the declared format"
        )
    return np.ascontiguousarray(array)


def _row_word(row: np.ndarray, bits_per_value: int, signed: bool) -> int:
    mask = (1 << bits_per_value) - 1
    word = 0
    offset = 0
    for raw in row.reshape(-1):
        code = int(raw)
        encoded = code & mask if signed else code
        word |= encoded << offset
        offset += bits_per_value
    return word


def _canonical_rows(
    candidates: np.ndarray, bits_per_value: int, signed: bool
) -> tuple[tuple[bytes, ...], tuple[int, ...]]:
    byte_width = max(1, (bits_per_value + 7) // 8)
    rows: list[bytes] = []
    words: list[int] = []
    for row in candidates:
        flat = row.reshape(-1)
        payload = b"".join(
            int(value).to_bytes(byte_width, "little", signed=signed) for value in flat
        )
        rows.append(payload)
        words.append(_row_word(row, bits_per_value, signed))
    return tuple(rows), tuple(words)


def empirical_candidate_loading_report(
    candidates: np.ndarray,
    bits_per_value: int,
    *,
    signed: bool = True,
    ancilla_qubits: int = 0,
    gate_type_count: int = 16,
    max_gate_arity: int = 2,
    exceptional_fraction: float = 0.01,
    max_minterm_table_bits: int = 4096,
) -> EmpiricalCandidateLoadingReport:
    """Audit an explicit quantized candidate table before coherent search.

    The first axis indexes candidates. Remaining axes define one candidate word.
    Duplicate rows are reported as observation fibres: under a uniform prior, a
    value lookup cannot recover the exact original index with probability above
    ``unique_candidate_count / candidate_count``.
    """

    array = _validate_codes(candidates, bits_per_value, signed)
    if ancilla_qubits < 0:
        raise ValueError("ancilla_qubits must be non-negative")
    if max_minterm_table_bits < 0:
        raise ValueError("max_minterm_table_bits must be non-negative")

    count = int(array.shape[0])
    candidate_shape = tuple(int(value) for value in array.shape[1:])
    values_per_candidate = int(np.prod(candidate_shape, dtype=np.int64))
    word_bits = values_per_candidate * int(bits_per_value)
    rows, words = _canonical_rows(array, int(bits_per_value), bool(signed))
    unique_rows = tuple(dict.fromkeys(rows))

    metadata = {
        "schema": "qrecon.empirical-candidates.v1",
        "candidate_count": count,
        "candidate_shape": candidate_shape,
        "bits_per_value": int(bits_per_value),
        "signed": bool(signed),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    for row in rows:
        digest.update(len(row).to_bytes(8, "little"))
        digest.update(row)

    explicit = explicit_lookup_description(
        count, word_bits, ancilla_qubits=ancilla_qubits
    )
    deduplicated = explicit_lookup_description(
        len(unique_rows), word_bits, ancilla_qubits=ancilla_qubits
    )
    typical = typical_lookup_circuit_lower_bound(
        count,
        word_bits,
        ancilla_qubits=ancilla_qubits,
        gate_type_count=gate_type_count,
        max_gate_arity=max_gate_arity,
        exceptional_fraction=exceptional_fraction,
    )

    if explicit.table_description_bits <= max_minterm_table_bits:
        minterm = lookup_minterm_resources(words, word_bits)
        skipped_reason = None
    else:
        minterm = None
        skipped_reason = (
            f"explicit table has {explicit.table_description_bits} bits, exceeding "
            f"max_minterm_table_bits={max_minterm_table_bits}"
        )

    return EmpiricalCandidateLoadingReport(
        candidate_count=count,
        unique_candidate_count=len(unique_rows),
        duplicate_candidate_count=count - len(unique_rows),
        candidate_shape=candidate_shape,
        values_per_candidate=values_per_candidate,
        bits_per_value=int(bits_per_value),
        signed=bool(signed),
        word_bits=word_bits,
        source_sha256=digest.hexdigest(),
        exact_index_bayes_success_uniform=len(unique_rows) / count,
        explicit_lookup=explicit,
        deduplicated_lookup=deduplicated,
        compiler_bit_probe_lower_bound=explicit_table_compiler_bit_probe_lower_bound(
            count, word_bits
        ),
        typical_circuit_lower_bound=typical,
        minterm_resources=minterm,
        minterm_skipped_reason=skipped_reason,
    )
