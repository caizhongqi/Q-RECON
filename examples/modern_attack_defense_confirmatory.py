from __future__ import annotations

import json
from typing import Mapping

from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    run_modern_timeseries_attack_suite,
    run_modern_timeseries_defense_suite,
    standard_modern_attack_variants,
    standard_modern_gradient_defenses,
)


ABSOLUTE_TOLERANCE_METRICS = {
    "1e-6": "within_1e-06_percent",
    "1e-5": "within_1e-05_percent",
    "1e-4": "within_0.0001_percent",
    "1e-3": "within_0.001_percent",
    "1e-2": "within_0.01_percent",
    "5e-2": "within_0.05_percent",
    "1e-1": "within_0.1_percent",
}
RELATIVE_L2_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5)


def _manifest() -> ModernTimeSeriesAttackManifest:
    return ModernTimeSeriesAttackManifest(
        dataset={
            "name": "gift_eval",
            "max_samples": 128,
            "context": 16,
            "horizon": 4,
            "streaming": True,
            "split": "train",
            "revision": "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0",
        },
        victim={
            "architecture": "patchtst",
            "patch_len": 4,
            "stride": 2,
            "padding_patch": True,
            "d_model": 4,
            "n_heads": 1,
            "e_layers": 1,
            "d_ff": 8,
            "dropout": 0.0,
            "head_dropout": 0.0,
            "revin": True,
        },
        training={
            "epochs": 1,
            "batch_size": 8,
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
        },
        attack={
            "prior": "direct",
            "bounded": True,
            "known_target": True,
            "steps": 150,
            "optimizer": "adam",
            "learning_rate": 0.03,
            # Release-aware inversion supports the same declared objective family as
            # GradientInversionAttack. The confirmatory attack matrix showed hybrid
            # had the best mean MSE among DLG/InvG/Q-RECON variants on this split.
            "match_mode": "hybrid",
            "layer_weighting": "parameter",
            "regularization": 0.0,
            "gradient_clip_norm": 10.0,
            "record_every": 25,
        },
        victim_seed=43,
        # 0..1 were exploratory, 2..21 were used by the paired learned study,
        # and 0..19 by the objective-family study. This is a disjoint confirmatory
        # record set while remaining inside the declared victim training data.
        attack_indices=tuple(range(40, 60)),
        attack_seeds=(101, 103, 107),
        attack_batch_size=1,
        exact_tolerance=0.1,
        relative_l2_threshold=0.5,
        confidence_level=0.95,
        bootstrap_samples=2000,
        bootstrap_seed=20260715,
        minimum_publication_batches=20,
        minimum_publication_attack_seeds=3,
        publication_mode=True,
    )


def _rate(successes: int, trials: int) -> dict[str, int | float]:
    return {
        "successes": int(successes),
        "trials": int(trials),
        "rate": 0.0 if trials == 0 else float(successes / trials),
    }


def _strict_success_audit(report: Mapping[str, object]) -> dict[str, object]:
    attempts = report["attempts"]
    selected = report["selected_attempt_indices"]
    if not isinstance(attempts, list) or not isinstance(selected, dict):
        raise TypeError("attack/defense report has an unexpected selected-attempt schema")

    variants: dict[str, object] = {}
    for name, raw_indices in sorted(selected.items()):
        if not isinstance(raw_indices, list):
            raise TypeError("selected attempt indices must be a list")
        metrics: list[Mapping[str, object]] = []
        for raw_index in raw_indices:
            attempt = attempts[int(raw_index)]
            if not isinstance(attempt, dict):
                raise TypeError("attempt entries must be dictionaries")
            aligned = attempt.get("aligned_batch")
            if not isinstance(aligned, dict):
                continue
            aligned_metrics = aligned.get("aligned_metrics")
            if not isinstance(aligned_metrics, dict):
                continue
            metrics.append(aligned_metrics)

        trials = len(metrics)
        bitwise = sum(
            float(item.get("bitwise_equal_percent", 0.0)) >= 100.0 - 1e-12
            for item in metrics
        )
        absolute = {
            tolerance: _rate(
                sum(float(item.get(metric_name, 0.0)) >= 100.0 - 1e-12 for item in metrics),
                trials,
            )
            for tolerance, metric_name in ABSOLUTE_TOLERANCE_METRICS.items()
        }
        relative = {
            f"{threshold:g}": _rate(
                sum(float(item["relative_l2_error"]) <= threshold for item in metrics),
                trials,
            )
            for threshold in RELATIVE_L2_THRESHOLDS
        }
        variants[str(name)] = {
            "selected_batches": trials,
            "bitwise_exact_batch_success": _rate(bitwise, trials),
            "all_values_within_absolute_tolerance": absolute,
            "relative_l2_success": relative,
            "mean_mse": (
                None
                if trials == 0
                else sum(float(item["mse"]) for item in metrics) / trials
            ),
            "mean_relative_l2_error": (
                None
                if trials == 0
                else sum(float(item["relative_l2_error"]) for item in metrics) / trials
            ),
            "mean_correlation": (
                None
                if trials == 0
                else sum(float(item["correlation"]) for item in metrics) / trials
            ),
        }

    return {
        "semantics": {
            "bitwise_exact_batch_success": (
                "every reconstructed scalar is bitwise equal to the private reference"
            ),
            "all_values_within_absolute_tolerance": (
                "every reconstructed scalar lies within the named absolute tolerance"
            ),
            "relative_l2_success": (
                "the selected batch has relative L2 error no larger than the named threshold"
            ),
            "legacy_exact_batch_success_warning": (
                "the underlying benchmark field exact_batch_success uses the manifest's "
                "declared tolerance (0.1 here); it is not bitwise exact recovery"
            ),
        },
        "variants": variants,
    }


def main() -> None:
    manifest = _manifest()
    attacks = run_modern_timeseries_attack_suite(
        manifest,
        standard_modern_attack_variants(period=4),
    )
    defenses = run_modern_timeseries_defense_suite(
        manifest,
        standard_modern_gradient_defenses(),
    )
    attack_payload = attacks.to_dict()
    defense_payload = defenses.to_dict()
    payload = {
        "schema_version": "qrecon.modern-attack-defense-confirmatory.v2",
        "confirmatory_split": {
            "attack_indices": list(range(40, 60)),
            "exploratory_or_previous_indices": list(range(22)),
            "declared_before_execution": True,
        },
        "attacks": attack_payload,
        "defenses": defense_payload,
        "strict_success_audit": {
            "attacks": _strict_success_audit(attack_payload),
            "defenses": _strict_success_audit(defense_payload),
        },
        "cross_report_checks": {
            "dataset_sha256_match": attacks.dataset_sha256 == defenses.dataset_sha256,
            "model_sha256_match": attacks.model_sha256 == defenses.model_sha256,
            "victim_class_match": attacks.victim_class == defenses.victim_class,
            "trainable_parameters_match": (
                attacks.trainable_parameters == defenses.trainable_parameters
            ),
        },
        "claim_boundary": (
            "This is a classical white-box, known-target reconstruction matrix on "
            "a modern PatchTST victim. The experiment compares matched optimization "
            "objectives and release mechanisms; it does not supply a coherent "
            "PatchTST oracle or establish quantum advantage. Strict success fields "
            "separate bitwise equality and named tolerances from the legacy 0.1 "
            "tolerance label."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    checks = payload["cross_report_checks"]
    if not attacks.quality_gate.passed:
        raise SystemExit("confirmatory attack-suite quality gate failed")
    if not defenses.quality_gate.passed:
        raise SystemExit("confirmatory defense-suite quality gate failed")
    if not all(checks.values()):
        raise SystemExit("attack/defense report identity check failed")


if __name__ == "__main__":
    main()
