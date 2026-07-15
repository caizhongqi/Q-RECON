from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

SOURCE_REPOSITORY = "zhouhaoyi/ETDataset"
SOURCE_COMMIT = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
SOURCE_PATH = "ETT-small/ETTm1.csv"
EXPECTED_GIT_BLOB_SHA1 = "62c9f979e2dc5b05d9cc5b38891e60e6baf77417"
SOURCE_URL = (
    "https://raw.githubusercontent.com/"
    f"{SOURCE_REPOSITORY}/{SOURCE_COMMIT}/{SOURCE_PATH}"
)


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the immutable ETTm1 CSV and verify its Git blob"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ETT-small/ETTm1.csv"),
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
        payload = response.read()
    observed_blob = git_blob_sha1(payload)
    if observed_blob != EXPECTED_GIT_BLOB_SHA1:
        raise RuntimeError(
            "ETTm1 Git blob mismatch: "
            f"expected {EXPECTED_GIT_BLOB_SHA1}, observed {observed_blob}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    report = {
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_path": SOURCE_PATH,
        "source_url": SOURCE_URL,
        "git_blob_sha1": observed_blob,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "output_path": str(args.output),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.manifest is None:
        print(encoded)
    else:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
