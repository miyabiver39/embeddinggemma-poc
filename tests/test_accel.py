"""推論の高速化(embedders/accel.py)のテスト。モデルを使わず、詰め物の除去の条件だけを確かめる。"""

from __future__ import annotations

import os
import types

import pytest

from mediasearch.config import Settings
from mediasearch.embedders import accel


class _FakeModel:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(vision_config=types.SimpleNamespace(pooling_kernel_size=3))
        self.received: list[tuple[int, int]] = []

    def get_image_features(self, pixel_values, image_position_ids=None, **kwargs):
        self.received.append((pixel_values.shape[1], image_position_ids.shape[1]))
        return "ok"


def _positions(valid: list[int], total: int):
    torch = pytest.importorskip("torch")
    pos = torch.full((len(valid), total, 2), -1, dtype=torch.long)
    for i, n in enumerate(valid):
        pos[i, :n] = torch.arange(n).unsqueeze(-1).repeat(1, 2)
    return pos


def _call(model, valid, total=36):
    torch = pytest.importorskip("torch")  # 開発用の環境(.[dev])には torch が無いため、その場合は飛ばす
    pixels = torch.zeros(len(valid), total, 4)
    return model.get_image_features(pixels, _positions(valid, total))


def test_padding_is_trimmed_when_all_images_have_the_same_size():
    m = _FakeModel()
    accel.trim_image_padding(m)
    assert _call(m, [27, 27]) == "ok"
    assert m.received[-1] == (27, 27)


@pytest.mark.parametrize(
    "valid",
    [
        [27, 18],  # 大きさの違う画像が混ざる → マスクつきのまま
        [26, 26],  # プーリング(3x3 = 9)で割り切れない
        [36, 36],  # 詰め物が無い
    ],
)
def test_padding_is_kept_otherwise(valid):
    m = _FakeModel()
    accel.trim_image_padding(m)
    _call(m, valid)
    assert m.received[-1] == (36, 36)


def test_compile_cache_goes_under_data_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    root = accel.setup_compile_cache(tmp_path)
    assert root == tmp_path / "cache" / "torch" and root.is_dir()
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"].startswith(str(root))
    assert os.environ["TRITON_CACHE_DIR"].startswith(str(root))


def test_torch_compile_setting(monkeypatch):
    monkeypatch.delenv("TORCH_COMPILE", raising=False)
    assert Settings.from_env().torch_compile == "auto"
    monkeypatch.setenv("TORCH_COMPILE", "OFF")
    assert Settings.from_env().torch_compile == "off"
    monkeypatch.setenv("TORCH_COMPILE", "maybe")
    with pytest.raises(ValueError):
        Settings.from_env()
