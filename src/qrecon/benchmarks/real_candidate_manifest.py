from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

DatasetName = Literal[
    "gift_eval",
    "community_forensics_small",
    "synthetic_forecasting",
]
FeatureSelection = Literal["prefix", "uniform_stride"]
AccessContract = Literal[
    "explicit_compilation",
    "physical_qram",
    "succinct_generator",
    "classical_only",
]
OverflowMode = Literal["raise", "saturate"]


@dataclass(frozen=True)
class CandidateQuantizationSpec:
    bits_per_value: int
    fractional_bits: int
    signed: bool = True
    overflow: OverflowMode = "saturate"

    def __post_init__(self) -> None:
        if self.bits_per_value <= 0 or self.bits_per_value > 62:
            raise ValueError("bits_per_value must lie in [1, 62]")
        if self.fractional_bits < 0:
            raise ValueError("fractional_bits must be non-negative")
        if self.overflow not in ("raise", "saturate"):
            raise ValueError("overflow must be 'raise' or 'saturate'")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RealBatchGradientManifest:
    """Hashable contract for a real-candidate, two-record gradient benchmark."""

    dataset: DatasetName
    dataset_revision: str | None = None
    split: str = "train"
    candidate_count: int = 32
    context: int = 16
    horizon: int = 1
    image_size: int = 8
    streaming: bool = True
    sampling: str = "datasets"
    seed: int = 17
    feature_count: int = 8
    feature_selection: FeatureSelection = "uniform_stride"
    target_coordinate: int = 0
    target_batch_indices: tuple[int, int] = (0, 1)
    quantizations: tuple[CandidateQuantizationSpec, ...] = (
        CandidateQuantizationSpec(8, 4, True, "saturate"),
    )
    model_seed: int = 29
    model_weights: tuple[int, ...] = ()
    model_bias: int = 0
    gradient_bits: int = 32
    target_success: float = 0.9
    bbht_growth_factor: float = 8.0 / 7.0
    bbht_attempts_per_stage: int = 1
    bbht_max_stages: int = 256
    max_exact_population: int = 4096
    max_exact_batches: int = 4096
    max_basis_verification_bits: int = 10
    max_minterm_table_bits: int = 4096
    access_contract: AccessContract = "explicit_compilation"
    reusable_instances: tuple[int, ...] = (1, 10, 100)
    expected_selected_source_sha256: str | None = None
    publication_mode: bool = False
    schema_version: str = "qrecon.real-batch-gradient-manifest.v1"

    def __post_init__(self) -> None:
        if self.dataset not in (
            "gift_eval",
            "community_forensics_small",
            "synthetic_forecasting",
        ):
            raise ValueError(f"unsupported dataset: {self.dataset!r}")
        if not self.split:
            raise ValueError("split must be non-empty")
        if self.candidate_count < 2:
            raise ValueError("candidate_count must be at least two")
        for name in ("context", "horizon", "image_size", "feature_count"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.target_coordinate < 0:
            raise ValueError("target_coordinate must be non-negative")
        if self.feature_selection not in ("prefix", "uniform_stride"):
            raise ValueError("unsupported feature_selection")

        pair = tuple(int(value) for value in self.target_batch_indices)
        if len(pair) != 2 or pair[0] < 0 or pair[0] >= pair[1]:
            raise ValueError("target_batch_indices must contain two increasing indices")
        if pair[1] >= self.candidate_count:
            raise ValueError("target_batch_indices exceed candidate_count")
        object.__setattr__(self, "target_batch_indices", pair)

        quantizations = tuple(self.quantizations)
        if not quantizations:
            raise ValueError("quantizations must be non-empty")
        if len(set(quantizations)) != len(quantizations):
            raise ValueError("quantizations must not contain duplicates")
        object.__setattr__(self, "quantizations", quantizations)

        weights = tuple(int(value) for value in self.model_weights)
        if weights and len(weights) != self.feature_count:
            raise ValueError("model_weights must be empty or match feature_count")
        if weights and not any(weights):
            raise ValueError("explicit model_weights must contain a non-zero value")
        object.__setattr__(self, "model_weights", weights)
        object.__setattr__(self, "model_bias", int(self.model_bias))

        if self.gradient_bits <= 1 or self.gradient_bits > 62:
            raise ValueError("gradient_bits must lie in [2, 62]")
        if not math.isfinite(self.target_success) or not 0.0 < self.target_success < 1.0:
            raise ValueError("target_success must lie strictly between zero and one")
        if (
            not math.isfinite(self.bbht_growth_factor)
            or not 1.0 < self.bbht_growth_factor < 4.0 / 3.0
        ):
            raise ValueError("bbht_growth_factor must lie strictly between 1 and 4/3")
        for name in (
            "bbht_attempts_per_stage",
            "bbht_max_stages",
            "max_exact_population",
            "max_exact_batches",
            "max_basis_verification_bits",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_minterm_table_bits < 0:
            raise ValueError("max_minterm_table_bits must be non-negative")
        if self.access_contract not in (
            "explicit_compilation",
            "physical_qram",
            "succinct_generator",
            "classical_only",
        ):
            raise ValueError("unsupported access_contract")

        workloads = tuple(int(value) for value in self.reusable_instances)
        if not workloads or any(value <= 0 for value in workloads):
            raise ValueError("reusable_instances must contain positive workloads")
        if len(set(workloads)) != len(workloads):
            raise ValueError("reusable_instances must be unique")
        object.__setattr__(self, "reusable_instances", workloads)

        expected = self.expected_selected_source_sha256
        if expected is not None:
            normalized = expected.lower()
            if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
                raise ValueError("expected_selected_source_sha256 must be a SHA256 hex digest")
            object.__setattr__(self, "expected_selected_source_sha256", normalized)

        if self.publication_mode and self.dataset != "synthetic_forecasting":
            revision = "" if self.dataset_revision is None else self.dataset_revision.strip()
            if revision.lower() in ("", "main", "master", "latest"):
                raise ValueError("publication_mode requires an immutable dataset revision")
            if self.expected_selected_source_sha256 is None:
                raise ValueError("publication_mode requires the selected-source SHA256")
        if (
            self.publication_mode
            and self.dataset == "community_forensics_small"
            and self.sampling == "api"
        ):
            raise ValueError("publication_mode forbids the unversioned rows API path")

    def resolved_model_weights(self) -> tuple[int, ...]:
        if self.model_weights:
            return self.model_weights
        rng = random.Random(int(self.model_seed))
        choices = (-2, -1, 1, 2)
        return tuple(rng.choice(choices) for _ in range(self.feature_count))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "candidate_count": self.candidate_count,
            "context": self.context,
            "horizon": self.horizon,
            "image_size": self.image_size,
            "streaming": self.streaming,
            "sampling": self.sampling,
            "seed": self.seed,
            "feature_count": self.feature_count,
            "feature_selection": self.feature_selection,
            "target_coordinate": self.target_coordinate,
            "target_batch_indices": list(self.target_batch_indices),
            "quantizations": [item.to_dict() for item in self.quantizations],
            "model_seed": self.model_seed,
            "model_weights": list(self.model_weights),
            "model_bias": self.model_bias,
            "gradient_bits": self.gradient_bits,
            "target_success": self.target_success,
            "bbht_growth_factor": self.bbht_growth_factor,
            "bbht_attempts_per_stage": self.bbht_attempts_per_stage,
            "bbht_max_stages": self.bbht_max_stages,
            "max_exact_population": self.max_exact_population,
            "max_exact_batches": self.max_exact_batches,
            "max_basis_verification_bits": self.max_basis_verification_bits,
            "max_minterm_table_bits": self.max_minterm_table_bits,
            "access_contract": self.access_contract,
            "reusable_instances": list(self.reusable_instances),
            "expected_selected_source_sha256": self.expected_selected_source_sha256,
            "publication_mode": self.publication_mode,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RealBatchGradientManifest":
        version = str(payload.get("schema_version", ""))
        expected = "qrecon.real-batch-gradient-manifest.v1"
        if version != expected:
            raise ValueError(f"unsupported manifest schema: {version!r}")
        raw_quantizations = payload.get("quantizations")
        if not isinstance(raw_quantizations, list):
            raise ValueError("quantizations must be a list")
        quantizations = tuple(
            CandidateQuantizationSpec(**dict(item))
            for item in raw_quantizations
            if isinstance(item, Mapping)
        )
        if len(quantizations) != len(raw_quantizations):
            raise ValueError("each quantization must be an object")
        values = dict(payload)
        values.pop("schema_version", None)
        values["quantizations"] = quantizations
        for name in ("target_batch_indices", "model_weights", "reusable_instances"):
            raw = values.get(name)
            if isinstance(raw, list):
                values[name] = tuple(int(value) for value in raw)
        return cls(**values)  # type: ignore[arg-type]

    @classmethod
    def from_json(cls, text: str) -> "RealBatchGradientManifest":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("manifest JSON must contain an object")
        return cls.from_dict(payload)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LoadedRealCandidateSet:
    manifest_sha256: str
    dataset: DatasetName
    dataset_revision: str | None
    split: str
    source_candidate_count: int
    source_input_shape: tuple[int, ...]
    source_target_shape: tuple[int, ...]
    feature_indices: tuple[int, ...]
    target_coordinate: int
    selected_source_sha256: str
    expected_source_sha256: str | None
    source_hash_matches: bool | None
    features: np.ndarray
    targets: np.ndarray

    def to_dict(self, *, include_values: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "manifest_sha256": self.manifest_sha256,
            "dataset": self.dataset,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "source_candidate_count": self.source_candidate_count,
            "source_input_shape": list(self.source_input_shape),
            "source_target_shape": list(self.source_target_shape),
            "feature_indices": list(self.feature_indices),
            "target_coordinate": self.target_coordinate,
            "selected_source_sha256": self.selected_source_sha256,
            "expected_source_sha256": self.expected_source_sha256,
            "source_hash_matches": self.source_hash_matches,
            "selected_feature_shape": list(self.features.shape),
            "selected_target_shape": list(self.targets.shape),
        }
        if include_values:
            payload["features"] = self.features.tolist()
            payload["targets"] = self.targets.tolist()
        return payload


CandidateLoader = Callable[[RealBatchGradientManifest], object]


def _as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("candidate tensors must be finite numeric arrays")
    return np.ascontiguousarray(array)


def _default_loader(manifest: RealBatchGradientManifest) -> object:
    from qrecon.data import (
        load_community_forensics,
        load_gifteval,
        synthetic_forecasting,
    )

    if manifest.dataset == "synthetic_forecasting":
        return synthetic_forecasting(
            samples=manifest.candidate_count,
            context=manifest.context,
            horizon=manifest.horizon,
            seed=manifest.seed,
        )
    if manifest.dataset == "gift_eval":
        return load_gifteval(
            max_series=manifest.candidate_count,
            context=manifest.context,
            horizon=manifest.horizon,
            streaming=manifest.streaming,
            split=manifest.split,
            revision=manifest.dataset_revision,
        )
    return load_community_forensics(
        max_images=manifest.candidate_count,
        image_size=manifest.image_size,
        streaming=manifest.streaming,
        seed=manifest.seed,
        sampling=manifest.sampling,
        split=manifest.split,
        revision=manifest.dataset_revision,
    )


def _extract_inputs_targets(dataset: object) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(dataset, "tensors"):
        tensors = tuple(getattr(dataset, "tensors"))
    elif isinstance(dataset, (tuple, list)):
        tensors = tuple(dataset)
    else:
        raise TypeError("candidate loader must return a TensorDataset or (inputs, targets)")
    if len(tensors) < 2:
        raise ValueError("candidate loader must return inputs and targets")
    return _as_numpy(tensors[0]), _as_numpy(tensors[1])


def _selected_source_hash(
    features: np.ndarray,
    targets: np.ndarray,
    feature_indices: tuple[int, ...],
    target_coordinate: int,
) -> str:
    metadata = {
        "schema": "qrecon.real-selected-candidates.v1",
        "feature_shape": tuple(int(value) for value in features.shape),
        "feature_dtype": features.dtype.str,
        "target_shape": tuple(int(value) for value in targets.shape),
        "target_dtype": targets.dtype.str,
        "feature_indices": feature_indices,
        "target_coordinate": int(target_coordinate),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(features).tobytes(order="C"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(targets).tobytes(order="C"))
    return digest.hexdigest()


def load_real_candidate_set(
    manifest: RealBatchGradientManifest,
    *,
    loader_overrides: Mapping[str, CandidateLoader] | None = None,
) -> LoadedRealCandidateSet:
    loader = None if loader_overrides is None else loader_overrides.get(manifest.dataset)
    dataset = _default_loader(manifest) if loader is None else loader(manifest)
    inputs, targets = _extract_inputs_targets(dataset)
    if inputs.ndim < 2:
        raise ValueError("inputs must have shape (candidate, ...)")
    if targets.ndim == 0:
        raise ValueError("targets must have a candidate axis")
    if inputs.shape[0] < manifest.candidate_count or targets.shape[0] < manifest.candidate_count:
        raise RuntimeError("loader returned fewer candidates than declared in the manifest")

    source_count = min(int(inputs.shape[0]), int(targets.shape[0]))
    inputs = np.ascontiguousarray(inputs[: manifest.candidate_count])
    targets = np.ascontiguousarray(targets[: manifest.candidate_count])
    flat_inputs = inputs.reshape(manifest.candidate_count, -1)
    flat_targets = targets.reshape(manifest.candidate_count, -1)
    if manifest.feature_count > flat_inputs.shape[1]:
        raise ValueError("feature_count exceeds the flattened input dimension")
    if manifest.target_coordinate >= flat_targets.shape[1]:
        raise ValueError("target_coordinate exceeds the flattened target dimension")

    if manifest.feature_selection == "prefix":
        indices = np.arange(manifest.feature_count, dtype=np.int64)
    else:
        indices = (
            np.arange(manifest.feature_count, dtype=np.int64)
            * flat_inputs.shape[1]
            // manifest.feature_count
        )
    if len(set(int(value) for value in indices)) != manifest.feature_count:
        raise RuntimeError("feature selection produced duplicate coordinates")

    selected_features = np.ascontiguousarray(flat_inputs[:, indices], dtype=np.float64)
    selected_targets = np.ascontiguousarray(
        flat_targets[:, manifest.target_coordinate : manifest.target_coordinate + 1],
        dtype=np.float64,
    )
    feature_indices = tuple(int(value) for value in indices)
    digest = _selected_source_hash(
        selected_features,
        selected_targets,
        feature_indices,
        manifest.target_coordinate,
    )
    expected = manifest.expected_selected_source_sha256
    matches = None if expected is None else digest == expected
    if manifest.publication_mode and matches is not True:
        raise RuntimeError("selected candidate source does not match the publication manifest")

    return LoadedRealCandidateSet(
        manifest_sha256=manifest.sha256,
        dataset=manifest.dataset,
        dataset_revision=manifest.dataset_revision,
        split=manifest.split,
        source_candidate_count=source_count,
        source_input_shape=tuple(int(value) for value in inputs.shape),
        source_target_shape=tuple(int(value) for value in targets.shape),
        feature_indices=feature_indices,
        target_coordinate=manifest.target_coordinate,
        selected_source_sha256=digest,
        expected_source_sha256=expected,
        source_hash_matches=matches,
        features=selected_features,
        targets=selected_targets,
    )
