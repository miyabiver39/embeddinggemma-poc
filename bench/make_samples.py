"""ベンチマーク用の合成の映像・音声・画像を作ります(実在の映像を使わずに、同じ条件で繰り返し計測するため)。

ffmpeg が必要です。作ったフォルダを、サーバーの INGEST_ROOTS の下(例: /recordings)にマウントして使います。

使い方:
    python bench/make_samples.py ./bench-media                # 既定: 1080p・60 秒(音声つき)、720p・60 秒、画像 50 枚
    python bench/make_samples.py ./bench-media --seconds 300  # 長い動画で計測する
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *cmd], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="ベンチマーク用の合成の映像・音声・画像を作ります")
    ap.add_argument("dest")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--images", type=int, default=50)
    args = ap.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg が見つかりません")
    dest = Path(args.dest)
    (dest / "images").mkdir(parents=True, exist_ok=True)
    s = args.seconds
    # testsrc2 は動きのある試験用の映像(H.264 の復号の負荷が、実際の録画に近くなるよう 30fps・GOP 2 秒)
    for name, size in (("1080p", "1920x1080"), ("720p", "1280x720")):
        out = dest / f"20260101_090000_{name}_{s}s.mp4"
        run(["-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration={s}",
             "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={s}",
             "-c:v", "libx264", "-preset", "veryfast", "-g", "60", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", str(out)])  # fmt: skip
        print(out)
    audio = dest / f"20260101_090000_audio_{s}s.wav"
    run(["-f", "lavfi", "-i", f"sine=frequency=330:sample_rate=16000:duration={s}", str(audio)])
    print(audio)
    for i in range(args.images):
        run(["-f", "lavfi", "-i", f"testsrc2=size=1920x1080:rate=1:duration=1,hue=h={i * 7}",
             "-frames:v", "1", "-q:v", "3", str(dest / "images" / f"img_{i:03d}.jpg")])  # fmt: skip
    print(f"{dest / 'images'}({args.images} 枚)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
