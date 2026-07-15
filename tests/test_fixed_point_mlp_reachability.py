import pytest

from qrecon.oracles import FixedPointFormat, QuantizedAffineLayer
from qrecon.oracles.fixed_point_mlp_reachability import (
    ReversibleFixedPointMLPValueOracle,
)


def _safe_model():
    input_format = FixedPointFormat(2, 0, True)
    hidden_format = FixedPointFormat(4, 0, True)
    hidden = QuantizedAffineLayer(
        weights=((1, -1), (1, 1)),
        biases=(0, 0),
        input_format=input_format,
        weight_format=FixedPointFormat(2, 0, True),
        bias_format=FixedPointFormat(4, 0, True),
        output_format=hidden_format,
        activation="relu",
    )
    output = QuantizedAffineLayer(
        weights=((1, 2), (-1, 1)),
        biases=(0, 1),
        input_format=hidden_format,
        weight_format=FixedPointFormat(3, 0, True),
        bias_format=FixedPointFormat(5, 0, True),
        output_format=FixedPointFormat(5, 0, True),
        activation="identity",
    )
    return hidden, output


def test_reachable_range_certificate_accepts_safe_composition_rejected_by_full_register_check():
    hidden, output = _safe_model()
    oracle = ReversibleFixedPointMLPValueOracle(hidden, output)

    certificate = oracle.reachability_certificate
    assert certificate.no_overflow
    assert certificate.hidden_encoded_bounds == ((0, 3), (0, 2))
    assert certificate.output_encoded_bounds == ((0, 7), (-3, 3))
    assert not oracle.output.requantizer.range_report.no_overflow
    assert oracle.verify_basis_permutation()


def test_reachable_range_certificate_still_rejects_a_truly_unsafe_network():
    fmt = FixedPointFormat(2, 0, True)
    hidden = QuantizedAffineLayer(
        weights=((1,),),
        biases=(0,),
        input_format=fmt,
        weight_format=fmt,
        bias_format=fmt,
        output_format=fmt,
        activation="relu",
    )
    output = QuantizedAffineLayer(
        weights=((1,),),
        biases=(1,),
        input_format=fmt,
        weight_format=fmt,
        bias_format=fmt,
        output_format=fmt,
        activation="identity",
    )
    with pytest.raises(OverflowError, match="reachable fixed-point MLP"):
        ReversibleFixedPointMLPValueOracle(hidden, output)


def test_non_strict_mode_exposes_failed_certificate_instead_of_claiming_safety():
    fmt = FixedPointFormat(2, 0, True)
    hidden = QuantizedAffineLayer(
        weights=((1,),),
        biases=(0,),
        input_format=fmt,
        weight_format=fmt,
        bias_format=fmt,
        output_format=fmt,
        activation="relu",
    )
    output = QuantizedAffineLayer(
        weights=((1,),),
        biases=(1,),
        input_format=fmt,
        weight_format=fmt,
        bias_format=fmt,
        output_format=fmt,
        activation="identity",
    )
    oracle = ReversibleFixedPointMLPValueOracle(
        hidden, output, require_no_overflow=False
    )
    assert not oracle.reachability_certificate.no_overflow
