import io
import sys
import types

import numpy as np
import pytest
from PIL import Image

from qrecon.data.images import load_community_forensics
from qrecon.data.time_series import load_gifteval


def test_gifteval_forwards_split_streaming_and_revision(monkeypatch):
    calls = []

    def fake_load_dataset(name, **kwargs):
        calls.append((name, kwargs))
        return [{"target": np.array([0.0, 1.0, 2.0], dtype=np.float32)}]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    dataset = load_gifteval(
        max_series=1,
        context=2,
        horizon=1,
        streaming=True,
        split="validation",
        revision="abc123",
    )
    assert dataset.tensors[0].shape == (1, 2)
    assert calls == [
        (
            "Salesforce/GiftEval",
            {"split": "validation", "streaming": True, "revision": "abc123"},
        )
    ]


def test_community_datasets_path_forwards_revision(monkeypatch):
    image = Image.new("RGB", (2, 2), color=(10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    rows = [
        {"label": 0, "image_data": {"bytes": buffer.getvalue()}, "nsfw_flag": False},
        {"label": 1, "image_data": {"bytes": buffer.getvalue()}, "nsfw_flag": False},
    ]
    calls = []

    def fake_load_dataset(name, **kwargs):
        calls.append((name, kwargs))
        return rows

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    dataset = load_community_forensics(
        max_images=2,
        image_size=2,
        sampling="datasets",
        split="train",
        revision="deadbeef",
    )
    assert dataset.tensors[0].shape == (2, 3, 2, 2)
    assert calls == [
        (
            "OwensLab/CommunityForensics-Small",
            {"split": "train", "streaming": True, "revision": "deadbeef"},
        )
    ]


def test_community_rows_api_rejects_immutable_revision():
    with pytest.raises(ValueError, match="not revision-pinned"):
        load_community_forensics(
            max_images=2,
            sampling="api",
            revision="deadbeef",
        )
