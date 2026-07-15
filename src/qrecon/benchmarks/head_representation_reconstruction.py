from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass

import torch

from qrecon.attacks import (
    HeadRepresentationInversionAttack,
    leak_gradients,
)
from qrecon.benchmarks.modern_timeseries_reconstruction import (
    ModernTimeSeriesAttackManifest,
)
from qrecon.benchmarks.statistics import (
    benchmark_environment_manifest,
    summarize_proportion,
    summarize_scalar,
)
from qrecon.experiment import (
    _build_model,
    _load_dataset,
    _prior,
    _seed_everything,
    _train,
)
from qrecon.metrics import permutation_invariant_batch_metrics


@dataclass(frozen=True)
class HeadRepresentationBenchmarkAttempt:
    batch_start: int
    restart_seed: int
    status: str
    seconds: float
    best_objective: float | None
    best_representation_loss: float | None
    best_step: int | None
    rank_one_relative_residual: float | None
    aligned_batch: dict[str, object] | None
    error_type: str | None = None
    error_message_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HeadRepresentationBenchmarkReport:
    manifest: ModernTimeSeriesAttackManifest
    report_sha256: str
    dataset_sha256: str
    model_sha256: str
    victim_class: str
    trainable_parameters: int
    attempts: tuple[HeadRepresentationBenchmarkAttempt, ...]
    selected_attempt_indices: tuple[int, ...]
    summary: dict[str, object]
    environment: dict[str, object]
    theorem_scope: str

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": self.manifest.to_dict(),
            "report_sha256": self.report_sha256,
            "dataset_sha256": self.dataset_sha256,
            "model_sha256": self.model_sha256,
            "victim_class": self.victim_class,
            "trainable_parameters": self.trainable_parameters,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "selected_attempt_indices": list(self.selected_attempt_indices),
            "summary": self.summary,
            "environment": self.environment,
            "theorem_scope": self.theorem_scope,
        }


def _hash_tensors(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _hash_model(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def run_head_representation_reconstruction_benchmark(
    manifest: ModernTimeSeriesAttackManifest,
) -> HeadRepresentationBenchmarkReport:
    """Evaluate the analytic final-head leakage plus first-order input inversion.

    The present exact theorem is restricted to attack batch size one and one
    effective input to the final shared Linear. The real GIFT-Eval PatchTST path is
    univariate and satisfies this condition. Multivariate PatchTST/iTransformer
    shared heads require a separate mixture-identifiability analysis.
    """

    if manifest.attack_batch_size != 1:
        raise ValueError("head-representation benchmark currently requires batch size one")
    _seed_everything(manifest.victim_seed)
    dataset, task, mode = _load_dataset(
        {"seed": manifest.victim_seed, "dataset": dict(manifest.dataset)}
    )
    if task != "forecasting" or mode != "timeseries":
        raise ValueError("head-representation benchmark requires forecasting data")
    x, targets = dataset.tensors
    if x.ndim == 3 and x.shape[2] != 1:
        architecture = str(manifest.victim.get("architecture", "")).lower()
        if architecture in {"patchtst", "itransformer"}:
            raise ValueError(
                "multivariate shared-head PatchTST/iTransformer has multiple effective "
                "head samples; exact analytic representation recovery is not valid"
            )

    model = _build_model(dataset, task, dict(manifest.victim))
    _train(model, dataset, task, dict(manifest.training))
    attempts: list[HeadRepresentationBenchmarkAttempt] = []
    selected_indices: list[int] = []
    selected: list[HeadRepresentationBenchmarkAttempt] = []
    attack_config = dict(manifest.attack)

    for batch_start in manifest.attack_indices:
        if batch_start >= len(x):
            raise ValueError(f"attack index {batch_start} exceeds dataset size {len(x)}")
        true_x = x[batch_start : batch_start + 1].clone()
        true_target = targets[batch_start : batch_start + 1].clone()
        observed = leak_gradients(model, true_x, true_target, "forecasting")
        group: list[int] = []
        for restart_seed in manifest.attack_seeds:
            started = time.perf_counter()
            try:
                _seed_everything(restart_seed)
                prior = _prior(tuple(true_x.shape), "timeseries", attack_config)
                attack = HeadRepresentationInversionAttack(
                    model,
                    observed,
                    prior,
                    mode="timeseries",
                    effective_samples=1,
                    steps=int(attack_config.get("head_steps", attack_config.get("steps", 300))),
                    learning_rate=float(
                        attack_config.get(
                            "head_learning_rate",
                            attack_config.get("learning_rate", 0.05),
                        )
                    ),
                    regularization=float(
                        attack_config.get(
                            "head_regularization",
                            attack_config.get("regularization", 0.0),
                        )
                    ),
                    gradient_clip_norm=(
                        None
                        if attack_config.get("gradient_clip_norm") is None
                        else float(attack_config["gradient_clip_norm"])
                    ),
                    record_every=(
                        None
                        if attack_config.get("record_every") is None
                        else int(attack_config["record_every"])
                    ),
                )
                result = attack.run()
                aligned = permutation_invariant_batch_metrics(
                    true_x,
                    result.reconstruction,
                    mode="timeseries",
                    tolerance=manifest.exact_tolerance,
                ).to_dict()
                attempt = HeadRepresentationBenchmarkAttempt(
                    batch_start=batch_start,
                    restart_seed=restart_seed,
                    status="success",
                    seconds=time.perf_counter() - started,
                    best_objective=result.best_objective,
                    best_representation_loss=result.best_representation_loss,
                    best_step=result.best_step,
                    rank_one_relative_residual=(
                        result.leakage.rank_one_relative_residual
                    ),
                    aligned_batch=aligned,
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                attempt = HeadRepresentationBenchmarkAttempt(
                    batch_start=batch_start,
                    restart_seed=restart_seed,
                    status="failed",
                    seconds=time.perf_counter() - started,
                    best_objective=None,
                    best_representation_loss=None,
                    best_step=None,
                    rank_one_relative_residual=None,
                    aligned_batch=None,
                    error_type=type(exc).__name__,
                    error_message_sha256=hashlib.sha256(
                        message.encode("utf-8")
                    ).hexdigest(),
                )
            group.append(len(attempts))
            attempts.append(attempt)

        successful = [
            index
            for index in group
            if attempts[index].status == "success"
            and attempts[index].best_objective is not None
        ]
        if successful:
            chosen = min(
                successful,
                key=lambda index: (
                    float(attempts[index].best_objective),
                    float(attempts[index].best_representation_loss),
                    attempts[index].restart_seed,
                ),
            )
            selected_indices.append(chosen)
            selected.append(attempts[chosen])

    exact_success = 0
    relative_l2_success = 0
    scalar_values: dict[str, list[float]] = {}
    for attempt in selected:
        assert attempt.aligned_batch is not None
        aligned = attempt.aligned_batch
        metrics = aligned["aligned_metrics"]
        assert isinstance(metrics, dict)
        exact_success += int(bool(aligned["exact_batch_within_tolerance"]))
        relative_l2_success += int(
            float(metrics["relative_l2_error"]) <= manifest.relative_l2_threshold
        )
        for name, value in metrics.items():
            scalar_values.setdefault(str(name), []).append(float(value))
        scalar_values.setdefault("best_objective", []).append(
            float(attempt.best_objective)
        )
        scalar_values.setdefault("best_representation_loss", []).append(
            float(attempt.best_representation_loss)
        )
        scalar_values.setdefault("rank_one_relative_residual", []).append(
            float(attempt.rank_one_relative_residual)
        )
        scalar_values.setdefault("attack_seconds", []).append(float(attempt.seconds))

    def scalar(name: str, values: list[float]) -> dict[str, object]:
        return summarize_scalar(
            values,
            confidence_level=manifest.confidence_level,
            bootstrap_samples=manifest.bootstrap_samples,
            bootstrap_seed=manifest.bootstrap_seed,
            label=f"head-representation:{name}",
        ).to_dict()

    failures = sum(attempt.status != "success" for attempt in attempts)
    trials = len(manifest.attack_indices)
    summary = {
        "declared_attack_batches": trials,
        "selected_successful_batches": len(selected),
        "total_restart_attempts": len(attempts),
        "failed_restart_attempts": failures,
        "restart_completion": summarize_proportion(
            len(attempts) - failures,
            len(attempts),
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "exact_batch_success": summarize_proportion(
            exact_success,
            trials,
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "relative_l2_success": summarize_proportion(
            relative_l2_success,
            trials,
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "scalar_summaries": {
            name: scalar(name, values)
            for name, values in sorted(scalar_values.items())
            if values
        },
    }
    dataset_hash = _hash_tensors(x, targets)
    model_hash = _hash_model(model)
    report_identity = hashlib.sha256(
        (
            manifest.sha256
            + dataset_hash
            + model_hash
            + "qrecon.head-representation.v1"
        ).encode("ascii")
    ).hexdigest()
    return HeadRepresentationBenchmarkReport(
        manifest=manifest,
        report_sha256=report_identity,
        dataset_sha256=dataset_hash,
        model_sha256=model_hash,
        victim_class=type(model).__name__,
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
        attempts=tuple(attempts),
        selected_attempt_indices=tuple(selected_indices),
        summary=summary,
        environment=benchmark_environment_manifest(),
        theorem_scope=(
            "Exact hidden-representation recovery assumes a nonzero final-head bias "
            "gradient and exactly one effective sample at the final biased Linear. "
            "Input recovery from that representation remains an optimization problem."
        ),
    )
