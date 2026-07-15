from __future__ import annotations

from dataclasses import asdict, dataclass

from .fixed_point_affine import ReversibleFixedPointAffineValueOracle
from .fixed_point_mlp import (
    FixedPointMLPLayout,
    ReversibleFixedPointAffineReLUValueOracle,
    ReversibleFixedPointMLPValueOracle as _LegacyFixedPointMLPValueOracle,
    _remap_gates,
)
from .models import NetworkRangeReport, QuantizedAffineLayer, QuantizedNetwork
from .reversible import ReversibleCircuit


@dataclass(frozen=True)
class FixedPointMLPReachabilityCertificate:
    """Layerwise proof that every reachable fixed-point value is representable."""

    hidden_raw_bounds: tuple[tuple[int, int], ...]
    hidden_encoded_bounds: tuple[tuple[int, int], ...]
    output_raw_bounds: tuple[tuple[int, int], ...]
    output_encoded_bounds: tuple[tuple[int, int], ...]
    no_overflow: bool

    @classmethod
    def from_network_report(
        cls, report: NetworkRangeReport
    ) -> "FixedPointMLPReachabilityCertificate":
        if len(report.layer_reports) != 2:
            raise ValueError("two-layer MLP certificate requires exactly two layer reports")
        hidden, output = report.layer_reports
        return cls(
            hidden_raw_bounds=hidden.raw_output_bounds,
            hidden_encoded_bounds=hidden.encoded_output_bounds,
            output_raw_bounds=output.raw_output_bounds,
            output_encoded_bounds=output.encoded_output_bounds,
            no_overflow=report.no_overflow,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ReversibleFixedPointMLPValueOracle(_LegacyFixedPointMLPValueOracle):
    """Two-layer value oracle certified on the network's reachable state space.

    A component affine accumulator can have a wider register range than the
    values that the preceding layer can actually produce. Requiring the entire
    accumulator register to fit the next output format rejects safe networks.
    This implementation first propagates exact interval bounds through the
    declared ``Affine-ReLU-Affine`` network. Only after that proof succeeds does
    it compile component requantizers in reachable-domain mode.

    The resulting circuit is still a total reversible permutation. Bit-exact
    agreement with the public fixed-point model is certified for every original
    input word, because all intermediate words reached from those inputs lie in
    the proven layer intervals.
    """

    def __init__(
        self,
        hidden_layer: QuantizedAffineLayer,
        output_layer: QuantizedAffineLayer,
        *,
        require_no_overflow: bool = True,
        max_enumeration_bits: int = 16,
    ) -> None:
        if hidden_layer.activation != "relu":
            raise ValueError("hidden layer must use ReLU")
        if output_layer.activation != "identity":
            raise ValueError("output layer must use identity activation")
        if hidden_layer.output_dimension != output_layer.input_dimension:
            raise ValueError("hidden and output layer dimensions do not match")
        if hidden_layer.output_format != output_layer.input_format:
            raise ValueError("hidden output and final input formats must match exactly")

        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.max_enumeration_bits = int(max_enumeration_bits)
        self.network = QuantizedNetwork((hidden_layer, output_layer), output_mode="raw")
        self.network_range_report = self.network.range_report()
        self.reachability_certificate = FixedPointMLPReachabilityCertificate.from_network_report(
            self.network_range_report
        )
        if require_no_overflow and not self.reachability_certificate.no_overflow:
            raise OverflowError(
                "reachable fixed-point MLP values exceed a declared layer format; "
                f"certificate={self.reachability_certificate.to_dict()}"
            )

        # The network-level certificate, rather than the full contents of each
        # intermediate register, is the relevant semantic contract here.
        self.hidden = ReversibleFixedPointAffineReLUValueOracle(
            hidden_layer,
            require_no_overflow=False,
            max_enumeration_bits=max_enumeration_bits,
        )
        self.output = ReversibleFixedPointAffineValueOracle(
            output_layer,
            require_no_overflow=False,
            max_enumeration_bits=max_enumeration_bits,
        )

        offset = self.hidden.input_bits
        output_wires = tuple(range(offset, offset + self.output.output_bits))
        offset += self.output.output_bits
        hidden_wires = tuple(range(offset, offset + self.hidden.output_bits))
        offset += self.hidden.output_bits
        hidden_work = tuple(
            range(offset, offset + len(self.hidden.layout.work_wires))
        )
        offset += len(hidden_work)
        output_work = tuple(
            range(offset, offset + len(self.output.layout.work_wires))
        )
        offset += len(output_work)
        self.layout = FixedPointMLPLayout(
            tuple(range(self.hidden.input_bits)),
            output_wires,
            hidden_wires,
            hidden_work,
            output_work,
        )
        self.circuit = ReversibleCircuit(offset)

        hidden_mapping: dict[int, int] = {}
        hidden_mapping.update(
            zip(self.hidden.layout.input_wires, self.layout.input_wires)
        )
        hidden_mapping.update(
            zip(self.hidden.layout.output_wires, hidden_wires)
        )
        hidden_mapping.update(
            zip(self.hidden.layout.work_wires, hidden_work)
        )
        hidden_gates = _remap_gates(self.hidden.circuit.gates, hidden_mapping)

        output_mapping: dict[int, int] = {}
        output_mapping.update(
            zip(self.output.layout.input_wires, hidden_wires)
        )
        output_mapping.update(
            zip(self.output.layout.output_wires, output_wires)
        )
        output_mapping.update(
            zip(self.output.layout.work_wires, output_work)
        )
        output_gates = _remap_gates(self.output.circuit.gates, output_mapping)

        self.circuit.extend(hidden_gates)
        self.circuit.extend(output_gates)
        self.circuit.append_inverse(hidden_gates)
