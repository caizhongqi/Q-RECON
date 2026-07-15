from __future__ import annotations

import json
from itertools import product

from qrecon.oracles.fixed_point import FixedPointFormat
from qrecon.oracles.fixed_point_deep_mlp import (
    ReversibleFixedPointDeepMLPEqualityOracle,
)
from qrecon.oracles.models import QuantizedAffineLayer, QuantizedNetwork
from qrecon.oracles.z3_deep_inversion import solve_fixed_point_deep_mlp_with_z3


def _model():
    input_format = FixedPointFormat(2, 0, True)
    weight_format = FixedPointFormat(2, 0, True)
    bias_format = FixedPointFormat(8, 0, True)
    hidden_one_format = FixedPointFormat(4, 0, True)
    hidden_two_format = FixedPointFormat(5, 0, True)
    output_format = FixedPointFormat(6, 0, True)
    layers = (
        QuantizedAffineLayer(
            weights=((1, -1), (-1, 1)),
            biases=(0, 1),
            input_format=input_format,
            weight_format=weight_format,
            bias_format=bias_format,
            output_format=hidden_one_format,
            activation="relu",
        ),
        QuantizedAffineLayer(
            weights=((1, 1), (1, -1)),
            biases=(0, 0),
            input_format=hidden_one_format,
            weight_format=weight_format,
            bias_format=bias_format,
            output_format=hidden_two_format,
            activation="relu",
        ),
        QuantizedAffineLayer(
            weights=((1, -1),),
            biases=(0,),
            input_format=hidden_two_format,
            weight_format=weight_format,
            bias_format=bias_format,
            output_format=output_format,
            activation="identity",
        ),
    )
    domain = tuple(range(input_format.min_code, input_format.max_code + 1))
    return layers, (domain, domain)


def main() -> None:
    layers, domains = _model()
    network = QuantizedNetwork(layers, output_mode="raw")
    private_record = (0, 0)
    target = network.evaluate_codes(private_record)
    expected = tuple(
        sorted(
            candidate
            for candidate in product(*domains)
            if network.evaluate_codes(candidate) == target
        )
    )
    report = solve_fixed_point_deep_mlp_with_z3(
        layers,
        target,
        domains=domains,
    )
    oracle = ReversibleFixedPointDeepMLPEqualityOracle(
        layers,
        target,
        max_enumeration_bits=network.input_bits,
    )
    encoded_expected = tuple(
        sorted(network.encode_input_codes(candidate) for candidate in expected)
    )
    payload = {
        "private_record": list(private_record),
        "target_codes": list(target),
        "z3_report": report.to_dict(),
        "reference_solutions": [list(solution) for solution in expected],
        "z3_matches_reference": report.solutions == expected,
        "oracle_marked_inputs": list(oracle.marked_inputs()),
        "oracle_matches_reference": oracle.marked_inputs() == encoded_expected,
        "oracle_basis_permutation_verified": oracle.verify_basis_permutation(),
        "oracle_resources": oracle.resource_estimate(
            phase_kickback=True
        ).to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
