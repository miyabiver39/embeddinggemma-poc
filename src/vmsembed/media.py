"""ffmpeg / ffprobe による映像・音声の加工。

モデルに渡す前の加工(フレーム抽出・16kHz モノラル化)は、すべてここで行います。
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .embedders.base import SAMPLE_RATE


class MediaError(RuntimeError):
    """ffmpeg / ffprobe の失敗。メッセージに原因(標準エラー)を含めます。"""


@dataclass(frozen=True)
class MediaInfo:
    duration_ms: int
    has_video: bool
    has_audio: bool


def _run(cmd: list[str], timeout: int = 120) -> bytes:
    if shutil.which(cmd[0]) is None:
        raise MediaError(f"{cmd[0]} が見つかりません。ffmpeg をインストールしてください。")
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{cmd[0]} が {timeout} 秒でタイムアウトしました") from exc
    if done.returncode != 0:
        tail = done.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
        raise MediaError(f"{cmd[0]} が失敗しました: {' / '.join(tail)}")
    return done.stdout


def probe(path: str | Path) -> MediaInfo:
    """長さと、映像・音声のトラックの有無を調べます。"""
    out = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    data = json.loads(out)
    streams = data.get("streams", [])
    duration = float(data.get("format", {}).get("duration") or 0.0)
    if duration <= 0:
        for s in streams:
            duration = max(duration, float(s.get("duration") or 0.0))
    return MediaInfo(
        duration_ms=int(duration * 1000),
        has_video=any(s.get("codec_type") == "video" for s in streams),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def extract_frame(path: str | Path, at_ms: int, max_side: int = 448) -> np.ndarray:
    """指定した時刻のフレームを1枚、RGB 配列で取り出します。"""
    out = _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{max(at_ms, 0) / 1000:.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-q:v",
            "2",
            "pipe:1",
        ]
    )
    if not out:
        raise MediaError(f"{at_ms}ms のフレームを取り出せませんでした(動画の末尾を超えた可能性)")
    image = Image.open(io.BytesIO(out)).convert("RGB")
    if max_side > 0:
        image.thumbnail((max_side, max_side))
    return np.asarray(image)


def extract_audio(path: str | Path, start_ms: int, duration_ms: int) -> np.ndarray:
    """指定区間の音声を、16kHz モノラルの float32 配列(-1.0〜1.0)で取り出します。"""
    out = _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{max(start_ms, 0) / 1000:.3f}",
            "-t",
            f"{duration_ms / 1000:.3f}",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "s16le",
            "pipe:1",
        ],
        timeout=300,
    )
    return np.frombuffer(out, dtype="<i2").astype(np.float32) / 32768.0


def make_thumbnail(path: str | Path, at_ms: int, out_path: Path, width: int = 320) -> None:
    """サムネイル(JPEG)を作ります。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(extract_frame(path, at_ms, max_side=width))
    image.save(out_path, format="JPEG", quality=80)


def format_time(ms: int) -> str:
    """ミリ秒を MM:SS(1時間以上は H:MM:SS)にします。窓の中の時刻ラベルに使います。"""
    total = max(ms, 0) // 1000
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
