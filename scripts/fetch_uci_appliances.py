from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/374/"
    "appliances%2Benergy%2Bprediction.zip"
)
CSV_MEMBER = "energydata_complete.csv"
DATASET_DOI = "10.24432/C5VC8G"
LICENSE = "CC BY 4.0"
EXPECTED_ARCHIVE_SHA256 = (
    "2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a"
)
EXPECTED_CSV_SHA256 = (
    "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/UCI-appliances")
    parser.add_argument("--manifest", default="outputs/uci-appliances-source.json")
    parser.add_argument(
        "--expected-zip-sha256", default=EXPECTED_ARCHIVE_SHA256
    )
    parser.add_argument(
        "--expected-csv-sha256", default=EXPECTED_CSV_SHA256
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "appliances-energy-prediction.zip"
    csv_path = output_dir / CSV_MEMBER

    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "Q-RECON reproducible dataset calibration"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, archive_path.open(
        "wb"
    ) as target:
        shutil.copyfileobj(response, target)

    archive_sha = _sha256(archive_path)
    if args.expected_zip_sha256 and archive_sha != args.expected_zip_sha256:
        raise RuntimeError(
            f"archive SHA256 mismatch: expected {args.expected_zip_sha256}, "
            f"observed {archive_sha}"
        )

    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        candidates = [name for name in names if Path(name).name == CSV_MEMBER]
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one {CSV_MEMBER!r} member, observed {candidates!r}"
            )
        with archive.open(candidates[0]) as source, csv_path.open("wb") as target:
            shutil.copyfileobj(source, target)

    csv_sha = _sha256(csv_path)
    if args.expected_csv_sha256 and csv_sha != args.expected_csv_sha256:
        raise RuntimeError(
            f"CSV SHA256 mismatch: expected {args.expected_csv_sha256}, "
            f"observed {csv_sha}"
        )

    payload = {
        "schema_version": "qrecon.uci-appliances-source.v1",
        "source_url": SOURCE_URL,
        "dataset_doi": DATASET_DOI,
        "license": LICENSE,
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha,
        "csv_path": str(csv_path),
        "csv_bytes": csv_path.stat().st_size,
        "csv_sha256": csv_sha,
        "csv_member": CSV_MEMBER,
        "expected_archive_sha256": args.expected_zip_sha256,
        "expected_csv_sha256": args.expected_csv_sha256,
        "locked": bool(args.expected_zip_sha256 and args.expected_csv_sha256),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
