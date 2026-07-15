from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .candidate_loading import (
    EmpiricalCandidateLoadingReport,
    empirical_candidate_loading_report,
)

OverflowMode = Literal["raise", "saturate"]


@dataclass(frozen=True)
class QuantizedCandidateAudit:
    candidate_count: int
    candidate_shape: tuple[int, ...]
    source_dtype: str
    source_sha256: str
    original_unique_candidate_count: int
    quantized_unique_candidate_count: int
    quantization_induced_collision_count: int
    bits_per_value: int
    fractional_bits: int
    signed: bool
    overflow: OverflowMode
    scale: int
    minimum_code: int
    maximum_code: int
    saturation_count: int
    quantization_mse: float
    maximum_absolute_error: float
    codes: np.ndarray
    loading: EmpiricalCandidateLoadingReport

    @property
    def exact_index_bayes_success_before_quantization_uniform(self) -> float:
        return self.original_unique_candidate_count / self.candidate_count

    @property
    def exact_index_bayes_success_after_quantization_uniform(self) -> float:
        return self.quantized_unique_candidate_count / self.candidate_count

    def to_dict(self, *, include_codes: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidate_count": self.candidate_count,
            "candidate_shape": list(self.candidate_shape),
            "source_dtype": self.source_dtype,
            "source_sha256": self.source_sha256,
            "original_unique_candidate_count": self.original_unique_candidate_count,
            "quantized_unique_candidate_count": self.quantized_unique_candidate_count,
            "quantization_induced_collision_count": self.quantization_induced_collision_count,
            "bits_per_value": self.bits_per_value,
            "fractional_bits": self.fractional_bits,
            "signed": self.signed,
            "overflow": self.overflow,
            "scale": self.scale,
            "minimum_code": self.minimum_code,
            "maximum_code": self.maximum_code,
            "saturation_count": self.saturation_count,
            "quantization_mse": self.quantization_mse,
            "maximum_absolute_error": self.maximum_absolute_error,
            "exact_index_bayes_success_before_quantization_uniform": (
                self.exact_index_bayes_success_before_quantization_uniform
            ),
            "exact_index_bayes_success_after_quantization_uniform": (
                self.exact_index_bayes_success_after_quantization_uniform
            ),
            "loading": self.loading.to_dict(),
        }
        if include_codes:
            payload["codes"] = self.codes.tolist()
        return payload


def _as_numpy(values: object) -> np.ndarray:
    if hasattr(values, "detach") and hasattr(values, "cpu") and hasattr(values, "numpy"):
        values = values.detach().cpu().numpy()  # type: ignore[union-attr]
    array = np.asarray(values)
    if array.ndim < 2:
        raise ValueError("values must have shape (candidate, ...)")
    if array.shape[0] <= 0:
        raise ValueError("values must contain at least one candidate")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("values must be numeric")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    return np.ascontiguousarray(array)


def _row_bytes(array: np.ndarray) -> tuple[bytes, ...]:
    contiguous = np.ascontiguousarray(array)
    return tuple(np.ascontiguousarray(row).tobytes(order="C") for row in contiguous)


def _source_hash(array: np.ndarray) -> str:
    metadata = {
        "schema": "qrecon.tensor-candidates.v1",
        "shape": tuple(int(value) for value in array.shape),
        "dtype": array.dtype.str,
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0.0, np.floor(values + 0.5), np.ceil(values - 0.5))


def audit_quantized_candidate_tensor(
    values: object,
    bits_per_value: int,
    fractional_bits: int,
    *,
    signed: bool = True,
    overflow: OverflowMode = "raise",
    ancilla_qubits: int = 0,
    gate_type_count: int = 16,
    max_gate_arity: int = 2,
    exceptional_fraction: float = 0.01,
    max_minterm_table_bits: int = 4096,
) -> QuantizedCandidateAudit:
    """Quantize tensor candidates and audit information loss and lookup cost.

    Codes use deterministic round-to-nearest with half ties away from zero. The
    function never silently wraps: out-of-range values either raise or saturate
    according to the declared policy.
    """

    array = _as_numpy(values)
    bits = int(bits_per_value)
    fractional = int(fractional_bits)
    if bits <= 0:
        raise ValueError("bits_per_value must be positive")
    if fractional < 0:
        raise ValueError("fractional_bits must be non-negative")
    if overflow not in ("raise", "saturate"):
        raise ValueError("overflow must be 'raise' or 'saturate'")
    if bits > 62:
        raise ValueError("bits_per_value must not exceed 62 for int64 audit codes")

    minimum = -(1 << (bits - 1)) if signed else 0
    maximum = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    scale = 1 << fractional
    converted = array.astype(np.float64, copy=False)
    scaled = converted * scale
    if not np.isfinite(scaled).all():
        raise OverflowError("scaled values are not finite")
    rounded = _round_half_away_from_zero(scaled)
    outside = (rounded < minimum) | (rounded > maximum)
    saturation_count = int(np.count_nonzero(outside))
    if saturation_count and overflow == "raise":
        observed_min = float(rounded.min())
        observed_max = float(rounded.max())
        raise OverflowError(
            f"quantized codes [{observed_min}, {observed_max}] exceed [{minimum}, {maximum}]"
        )
    if overflow == "saturate":
        rounded = np.clip(rounded, minimum, maximum)
    codes = np.ascontiguousarray(rounded.astype(np.int64))
    reconstructed = codes.astype(np.float64) / scale
    error = reconstructed - converted

    original_unique = len(set(_row_bytes(array)))
    quantized_unique = len(set(_row_bytes(codes)))
    loading = empirical_candidate_loading_report(
        codes,
        bits,
        signed=signed,
        ancilla_qubits=ancilla_qubits,
        gate_type_count=gate_type_count,
        max_gate_arity=max_gate_arity,
        exceptional_fraction=exceptional_fraction,
        max_minterm_table_bits=max_minterm_table_bits,
    )
    if loading.unique_candidate_count != quantized_unique:
        raise RuntimeError("candidate loading audit disagrees with quantized row count")

    return QuantizedCandidateAudit(
        candidate_count=int(array.shape[0]),
        candidate_shape=tuple(int(value) for value in array.shape[1:]),
        source_dtype=array.dtype.str,
        source_sha256=_source_hash(array),
        original_unique_candidate_count=original_unique,
        quantized_unique_candidate_count=quantized_unique,
        quantization_induced_collision_count=max(0, original_unique - quantized_unique),
        bits_per_value=bits,
        fractional_bits=fractional,
        signed=bool(signed),
        overflow=overflow,
        scale=scale,
        minimum_code=minimum,
        maximum_code=maximum,
        saturation_count=saturation_count,
        quantization_mse=float(np.mean(error * error)),
        maximum_absolute_error=float(np.max(np.abs(error))),
        codes=codes,
        loading=loading,
    )
