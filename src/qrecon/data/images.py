from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from torch.utils.data import TensorDataset


def _decode_image(value: object) -> Image.Image:
    if isinstance(value, dict) and value.get("bytes") is not None:
        value = value["bytes"]
    if isinstance(value, str):
        value = value.strip('"')
        value = base64.b64decode(value)
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"unsupported image payload: {type(value)!r}")
    return Image.open(io.BytesIO(value)).convert("RGB")


def _to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_community_forensics(
    max_images: int = 128,
    image_size: int = 32,
    streaming: bool = True,
    exclude_nsfw: bool = True,
    seed: int = 17,
    sampling: str = "api",
    real_offset: int = 9000,
    split: str = "train",
    revision: str | None = None,
) -> TensorDataset:
    """Load the redistributable Community Forensics Small set.

    The lightweight rows-API path is intentionally treated as unversioned. A
    revision-pinned publication manifest must use ``sampling='datasets'`` so the
    requested commit SHA is forwarded to Hugging Face Datasets.
    """
    if sampling == "api":
        if split != "train":
            raise ValueError("the rows API adapter currently supports only split='train'")
        if revision is not None and revision.lower() not in ("", "main", "master"):
            raise ValueError(
                "the rows API path is not revision-pinned; use sampling='datasets'"
            )
        rows = _balanced_community_rows(
            max_images=max_images, seed=seed, real_offset=real_offset
        )
    elif sampling == "datasets":
        from datasets import load_dataset

        kwargs: dict[str, object] = {
            "split": split,
            "streaming": streaming,
        }
        if revision is not None:
            kwargs["revision"] = revision
        rows = load_dataset(
            "OwensLab/CommunityForensics-Small",
            **kwargs,
        )
    else:
        raise ValueError("sampling must be 'api' or 'datasets'")

    images: list[torch.Tensor] = []
    labels: list[int] = []
    class_limit = (max_images + 1) // 2
    class_counts = {0: 0, 1: 0}
    for row in rows:
        if exclude_nsfw and bool(row.get("nsfw_flag", False)):
            continue
        label = int(row["label"])
        if label in class_counts and class_counts[label] >= class_limit:
            continue
        try:
            images.append(_to_tensor(_decode_image(row["image_data"]), image_size))
        except Exception:
            continue
        labels.append(label)
        class_counts[label] = class_counts.get(label, 0) + 1
        if len(images) >= max_images:
            break
    if not images:
        raise RuntimeError("no usable Community Forensics images were found")
    return TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))


def _community_api_rows(offset: int, length: int) -> tuple[list[dict], int]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(
                "https://datasets-server.huggingface.co/rows",
                params={
                    "dataset": "OwensLab/CommunityForensics-Small",
                    "config": "default",
                    "split": "train",
                    "offset": offset,
                    "length": length,
                },
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            return [entry["row"] for entry in payload["rows"]], int(payload["num_rows_total"])
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Community Forensics API failed at offset {offset}") from last_error


def _row_range(offset: int, count: int, chunk_size: int = 8) -> list[dict]:
    result: list[dict] = []
    while len(result) < count:
        rows, _ = _community_api_rows(
            offset + len(result), min(chunk_size, count - len(result))
        )
        if not rows:
            break
        result.extend(rows)
    return result


def _balanced_community_rows(max_images: int, seed: int, real_offset: int) -> list[dict]:
    """Fetch balanced ranges without downloading the full 10k-image parquet.

    The fixed 2025 small benchmark is label-sorted. The default offset is pinned
    in the experiment config and can be updated if a future revision reorders it.
    """
    per_class = (max_images + 1) // 2
    left = _row_range(0, per_class * 2)
    right = _row_range(real_offset, per_class * 2)
    rng = np.random.default_rng(seed)
    rng.shuffle(left)
    rng.shuffle(right)
    return left[:per_class] + right[: max_images - per_class]


def load_image_folder(root: str | Path, image_size: int = 32) -> TensorDataset:
    """Load a CLOFAI export or any class-per-directory image dataset."""
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(str(root))
    images, labels = [], []
    for image, label in dataset:
        images.append(_to_tensor(image.convert("RGB"), image_size))
        labels.append(label)
    return TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))
