import numpy as np
import pytest

from qrecon.benchmarks.real_candidate_manifest import (
    CandidateQuantizationSpec,
    RealBatchGradientManifest,
    load_real_candidate_set,
)


def test_manifest_roundtrip_and_hash_are_canonical():
    manifest = RealBatchGradientManifest(
        dataset="synthetic_forecasting",
        candidate_count=4,
        context=6,
        horizon=2,
        feature_count=3,
        target_batch_indices=(0, 2),
        quantizations=(
            CandidateQuantizationSpec(6, 2, True, "raise"),
            CandidateQuantizationSpec(8, 4, True, "saturate"),
        ),
        model_weights=(1, -1, 2),
    )
    restored = RealBatchGradientManifest.from_json(manifest.canonical_json())
    assert restored == manifest
    assert restored.sha256 == manifest.sha256


def test_publication_mode_requires_revision_hash_and_versioned_community_path():
    with pytest.raises(ValueError, match="immutable dataset revision"):
        RealBatchGradientManifest(
            dataset="gift_eval",
            publication_mode=True,
        )
    with pytest.raises(ValueError, match="selected-source SHA256"):
        RealBatchGradientManifest(
            dataset="gift_eval",
            dataset_revision="0123456789abcdef",
            publication_mode=True,
        )
    with pytest.raises(ValueError, match="rows API"):
        RealBatchGradientManifest(
            dataset="community_forensics_small",
            dataset_revision="0123456789abcdef",
            expected_selected_source_sha256="0" * 64,
            sampling="api",
            publication_mode=True,
        )


def test_loader_override_selects_declared_features_and_target_coordinate():
    manifest = RealBatchGradientManifest(
        dataset="synthetic_forecasting",
        candidate_count=4,
        feature_count=3,
        feature_selection="uniform_stride",
        target_coordinate=1,
        target_batch_indices=(1, 3),
        quantizations=(CandidateQuantizationSpec(8, 2),),
        model_weights=(1, 1, 1),
    )
    inputs = np.arange(4 * 8, dtype=np.float32).reshape(4, 2, 4)
    targets = np.arange(4 * 2, dtype=np.float32).reshape(4, 2)

    loaded = load_real_candidate_set(
        manifest,
        loader_overrides={
            "synthetic_forecasting": lambda _: (inputs, targets),
        },
    )
    assert loaded.feature_indices == (0, 2, 5)
    assert loaded.features.tolist() == inputs.reshape(4, -1)[:, [0, 2, 5]].tolist()
    assert loaded.targets.reshape(-1).tolist() == targets[:, 1].tolist()
    assert loaded.source_hash_matches is None
    assert len(loaded.selected_source_sha256) == 64


def test_publication_source_hash_is_enforced_at_load_time():
    calibration = RealBatchGradientManifest(
        dataset="synthetic_forecasting",
        candidate_count=3,
        feature_count=1,
        target_batch_indices=(0, 1),
        model_weights=(1,),
    )
    values = (
        np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        np.zeros((3, 1), dtype=np.float32),
    )
    loaded = load_real_candidate_set(
        calibration,
        loader_overrides={"synthetic_forecasting": lambda _: values},
    )
    locked = RealBatchGradientManifest(
        dataset="synthetic_forecasting",
        candidate_count=3,
        feature_count=1,
        target_batch_indices=(0, 1),
        model_weights=(1,),
        expected_selected_source_sha256=loaded.selected_source_sha256,
        publication_mode=True,
    )
    verified = load_real_candidate_set(
        locked,
        loader_overrides={"synthetic_forecasting": lambda _: values},
    )
    assert verified.source_hash_matches is True
