from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

SOURCE_REPOSITORY = "zhouhaoyi/ETDataset"
SOURCE_COMMIT = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
VARIANTS = {
    "ettm1": {
        "path": "ETT-small/ETTm1.csv",
        "git_blob_sha1": "62c9f979e2dc5b05d9cc5b38891e60e6baf77417",
    },
    "ettm2": {
        "path": "ETT-small/ETTm2.csv",
        "git_blob_sha1": "f9d46027c987d442d2d8f81308d1d8c4d1d35db8",
    },
    "etth1": {
        "path": "ETT-small/ETTh1.csv",
        "git_blob_sha1": "a52c4925778c07c1ef1a2cf6fd01594919717d9e",
    },
    "etth2": {
        "path": "ETT-small/ETTh2.csv",
        "git_blob_sha1": "6e3760778598fdee5208213d3fce122637307eec",
    },
}


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def fetch_variant(name: str, output: Path) -> dict[str, object]:
    normalized = str(name).lower()
    if normalized not in VARIANTS:
        raise ValueError(f"unknown ETT variant {name!r}; choose from {sorted(VARIANTS)}")
    declaration = VARIANTS[normalized]
    source_path = str(declaration["path"])
    source_url = (
        "https://raw.githubusercontent.com/"
        f"{SOURCE_REPOSITORY}/{SOURCE_COMMIT}/{source_path}"
    )
    with urllib.request.urlopen(source_url, timeout=180) as response:
        payload = response.read()
    observed_blob = git_blob_sha1(payload)
    expected_blob = str(declaration["git_blob_sha1"])
    if observed_blob != expected_blob:
        raise RuntimeError(
            f"{normalized} Git blob mismatch: expected {expected_blob}, "
            f"observed {observed_blob}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return {
        "variant": normalized,
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_path": source_path,
        "source_url": source_url,
        "git_blob_sha1": observed_blob,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "output_path": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download one immutable ETT variant and verify its Git blob"
    )
    parser.add_argument("variant", choices=tuple(sorted(VARIANTS)))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report = fetch_variant(args.variant, args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
