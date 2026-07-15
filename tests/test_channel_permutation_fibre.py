from __future__ import annotations

import pytest

from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    run_channel_permutation_fibre_benchmark,
)
from qrecon.theory import (
    channel_permutation_fibre_bound,
    channel_permutation_orbit_size,
)


def test_multiset_channel_orbit_size_and_uniform_ceiling():
    assert channel_permutation_orbit_size((1, 1, 1)) == 6
    assert channel_permutation_orbit_size((2, 1)) == 3
    report = channel_permutation_fibre_bound(("a", "b", "a", "c"))
    assert report.channels == 4
    assert report.multiplicities == (2, 1, 1)
    assert report.orbit_size == 12
    assert report.uniform_exact_ordered_recovery_ceiling == pytest.approx(1 / 12)
    with pytest.raises(ValueError):
        channel_permutation_orbit_size(())
    with pytest.raises(ValueError):
        channel_permutation_orbit_size((1, 0))


def test_itransformer_full_gradient_is_invariant_on_channel_generators():
    manifest = ModernTimeSeriesAttackManifest(
        dataset={
            "name": "synthetic_multivariate_time",
            "max_samples": 4,
            "context": 4,
            "horizon": 2,
            "channels": 4,
        },
        victim={
            "architecture": "itransformer",
            "d_model": 4,
            "n_heads": 1,
            "e_layers": 1,
            "d_ff": 8,
            "dropout": 0.0,
            "revin": False,
        },
        training={
            "epochs": 1,
            "batch_size": 2,
            "optimizer": "adam",
            "learning_rate": 1e-3,
        },
        attack={"prior": "direct", "known_target": True, "steps": 1},
        victim_seed=23,
        attack_indices=(0,),
        attack_seeds=(101,),
        attack_batch_size=1,
        bootstrap_samples=20,
        publication_mode=False,
    )
    report = run_channel_permutation_fibre_benchmark(manifest, tolerance=1e-5)
    point = report.points[0]
    assert point.channels == 4
    assert point.generator_count == 3
    assert point.orbit_size > 1
    assert point.all_generator_checks_pass
    assert point.maximum_output_equivariance_error < 1e-5
    assert point.maximum_gradient_invariance_error < 1e-5
    assert point.maximum_gradient_relative_error < 1e-5


def test_permutation_benchmark_rejects_channel_identity_parameters():
    manifest = ModernTimeSeriesAttackManifest(
        dataset={
            "name": "synthetic_multivariate_time",
            "max_samples": 4,
            "context": 4,
            "horizon": 2,
            "channels": 3,
        },
        victim={
            "architecture": "itransformer",
            "d_model": 4,
            "n_heads": 1,
            "e_layers": 1,
            "d_ff": 8,
            "dropout": 0.0,
            "revin": True,
        },
        training={
            "epochs": 1,
            "batch_size": 2,
            "optimizer": "adam",
            "learning_rate": 1e-3,
        },
        attack={"prior": "direct", "known_target": True, "steps": 1},
        victim_seed=23,
        attack_indices=(0,),
        attack_seeds=(101,),
        attack_batch_size=1,
        bootstrap_samples=20,
        publication_mode=False,
    )
    with pytest.raises(ValueError, match="revin=false"):
        run_channel_permutation_fibre_benchmark(manifest)
