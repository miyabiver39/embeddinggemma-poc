"""デバイス判定の子プロセス確認(PyTorch 不要)。"""

from __future__ import annotations

from mediasearch.embedders.local import run_probe


def test_probe_returns_last_line():
    assert run_probe("print('warn'); print(2)") == "2"


def test_probe_survives_segfault():
    # :intel イメージで GPU が無いと torch.xpu の判定がセグメンテーション違反で落ちる。
    # 子プロセスが落ちても、呼び出し側は None を受け取って CPU に切り替えられること
    assert run_probe("import os, signal; os.kill(os.getpid(), signal.SIGSEGV)") is None


def test_probe_timeout():
    assert run_probe("import time; time.sleep(5)", timeout=0.5) is None
