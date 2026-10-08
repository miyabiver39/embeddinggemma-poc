"""窓の切り出し計算とトークン予算の単体テスト。"""

from __future__ import annotations

import pytest

from vmsembed.config import Settings
from vmsembed.pipeline import IngestParams, plan_windows


def params(**kw) -> IngestParams:
    return IngestParams.from_settings(Settings.from_env(), **kw)


def plan(duration_ms, window_sec=2, frames=2, overlap=0):
    return plan_windows(duration_ms, window_sec, frames, overlap)


def test_plan_windows_basic():
    wins = plan(10_000)
    assert wins[0].start_ms == 0 and wins[-1].end_ms <= 10_000
    assert len(wins) == 5


def test_overlap_makes_more_windows():
    a = plan(10_000)
    b = plan(10_000, overlap=1)
    assert len(b) > len(a)


def test_token_budget_rejected():
    with pytest.raises(ValueError):
        params(frames_per_window=40)
