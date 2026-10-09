"""窓の切り出し計算とトークン予算の単体テスト。"""

from __future__ import annotations

import pytest

from mediasearch.config import Settings
from mediasearch.pipeline import IngestParams, decode_step_ms, plan_windows


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


def test_samples_are_centers_of_sub_intervals():
    wins = plan(10_000)
    assert wins[0].sample_ms == [500, 1500] and wins[1].sample_ms == [2500, 3500]
    assert all(t < 10_000 for w in wins for t in w.sample_ms)  # 末尾ちょうどは使わない
    assert plan(10_000, window_sec=1, frames=1)[0].sample_ms == [500]


def test_decode_step_divides_all_samples():
    wins = plan(60_000)
    step = decode_step_ms(wins)
    assert step == 500 and all(t % step == 0 for w in wins for t in w.sample_ms)
    assert decode_step_ms(plan(10_000, window_sec=3, frames=3)) >= 100  # 割り切れなくても下限を守る
