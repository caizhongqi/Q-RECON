from __future__ import annotations

import json

import torch

from qrecon.theory import (
    certify_head_representation_perturbation,
    combine_head_perturbation_bounds,
    common_scale_invariance_error,
    gaussian_head_bounds,
    recover_head_representation,
    uniform_quantization_head_bounds,
)


def main() -> None:
    bias = torch.tensor([0.75, -0.5, 0.25], dtype=torch.float64)
    feature = torch.tensor([1.25, -0.5, 2.0, 0.75], dtype=torch.float64)
    weight = bias.unsqueeze(1) * feature.unsqueeze(0)
    scale = 0.08

    quantization = uniform_quantization_head_bounds(
        output_dimension=bias.numel(),
        feature_dimension=feature.numel(),
        step=2e-3,
    )
    gaussian = gaussian_head_bounds(
        output_dimension=bias.numel(),
        feature_dimension=feature.numel(),
        noise_std=1e-4,
        failure_probability=0.01,
    )
    combined = combine_head_perturbation_bounds(
        quantization,
        gaussian,
        provenance="int-like quantization plus Gaussian release noise",
    )

    observed_bias = scale * bias
    observed_weight = scale * weight
    certificate = certify_head_representation_perturbation(
        observed_weight,
        observed_bias,
        combined,
    )
    recovered = recover_head_representation(observed_weight, observed_bias)
    print(
        json.dumps(
            {
                "clean_feature": feature.tolist(),
                "scaled_feature": recovered.tolist(),
                "common_scale_invariance_l2_error": common_scale_invariance_error(
                    weight, bias, scale
                ),
                "perturbation_certificate": certificate.to_dict(),
                "claim_boundary": (
                    "Common nonzero global clipping does not remove one-effective-sample "
                    "final-head ratio leakage. Noise and quantization require explicit "
                    "perturbation bounds, and representation recovery does not by itself "
                    "guarantee exact input inversion."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
