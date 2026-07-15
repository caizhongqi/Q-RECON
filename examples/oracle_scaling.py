from __future__ import annotations

import json

from qrecon.oracles import TruthTableOracle, compare_exact_syntheses


def _resource_row(name: str, input_bits: int, function) -> dict[str, object]:
    oracle = TruthTableOracle.from_function(input_bits, 1, function, max_input_bits=input_bits)
    comparison = compare_exact_syntheses(oracle)
    return {
        "family": name,
        "input_bits": input_bits,
        "marked": len(oracle.marked_inputs()),
        "selected": comparison.selected,
        "minterm": comparison.minterm.to_dict(),
        "anf": comparison.anf.to_dict(),
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    for input_bits in range(2, 11):
        rows.append(
            _resource_row(
                "parity",
                input_bits,
                lambda word: word.bit_count() & 1,
            )
        )
        rows.append(
            _resource_row(
                "all_zero_equality",
                input_bits,
                lambda word: int(word == 0),
            )
        )
        rows.append(
            _resource_row(
                "majority",
                input_bits,
                lambda word, width=input_bits: int(word.bit_count() > width // 2),
            )
        )
    print(json.dumps({"rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
