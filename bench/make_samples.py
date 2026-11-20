"""ベンチマーク用の合成の映像・音声・画像を作ります(実在の映像を使わずに、同じ条件で繰り返し計測するため)。

ffmpeg が必要です。作ったフォルダを、サーバーの INGEST_ROOTS の下(例: /recordings)にマウントして使います。

使い方:
    python bench/make_samples.py ./bench-media                # 既定: 1080p・60 秒(音声つき)、720p・60 秒、画像 50 枚
    python bench/make_samples.py ./bench-media --seconds 300  # 長い動画で計測する
    python bench/make_samples.py ./bench-media --split-set    # 共通ベンチマーク(MATRIX.md)の素材だけを作る
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *cmd], check=True)


# 共通ベンチマーク(MATRIX.md)の素材
SPLIT_TOTAL_SEC = 480  # 元の動画の長さ。1 / 2 / 4 / 8 本に分けても、60 秒の倍数になる
SPLIT_COUNTS = (1, 2, 4, 8)
X264 = ["-c:v", "libx264", "-preset", "veryfast", "-g", "60", "-keyint_min", "60", "-sc_threshold", "0",
        "-pix_fmt", "yuv420p"]  # fmt: skip
QUALITY_SIDES = (448, 336, 224)  # IMAGE_MAX_SIDE の条件。compute の API は縮小しないため、送る側で縮小しておく
# 品質の評価用の単色の画像と、その正解の文章(日本語と英語)
QUALITY_COLORS = [
    ("red", "#e00000", "真っ赤な画面", "a solid red screen"),
    ("blue", "#0030e0", "真っ青な画面", "a solid blue screen"),
    ("green", "#00b000", "緑一色の画面", "a solid green screen"),
    ("yellow", "#f0e000", "黄色一色の画面", "a solid yellow screen"),
    ("white", "#ffffff", "真っ白な画面", "a solid white screen"),
    ("black", "#000000", "真っ黒な画面", "a solid black screen"),
    ("orange", "#ff8000", "オレンジ色一色の画面", "a solid orange screen"),
    ("purple", "#8000c0", "紫一色の画面", "a solid purple screen"),
    ("pink", "#ff80c0", "ピンク色一色の画面", "a solid pink screen"),
    ("cyan", "#00e0e0", "水色一色の画面", "a solid cyan screen"),
    ("gray", "#808080", "灰色一色の画面", "a solid gray screen"),
    ("brown", "#704020", "茶色一色の画面", "a solid brown screen"),
]  # fmt: skip


def make_split_set(dest: Path) -> None:
    """480 秒の元の動画を 1 本作り、1 / 2 / 4 / 8 本に分けます(中身と合計の長さは、どの分け方でも同じ)。

    分けたファイルは、元の動画から同じ設定でエンコードし直して作ります(フレームの数は、どの分け方でも 14400)。
    """
    split = dest / "split"
    split.mkdir(parents=True, exist_ok=True)
    master = split / "master_480s.mp4"
    if not master.exists():
        run(["-f", "lavfi", "-i", f"testsrc2=size=1920x1080:rate=30:duration={SPLIT_TOTAL_SEC}",
             *X264, "-an", str(master)])  # fmt: skip
    for k in SPLIT_COUNTS:
        out = split / f"k{k}"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir()
        part = SPLIT_TOTAL_SEC // k
        for i in range(k):
            # ファイル名の日時は、元の動画の中での位置に合わせる(09:00:00 から part 秒ずつ)
            start = i * part
            hh, mm, ss = 9 + start // 3600, (start % 3600) // 60, start % 60
            name = out / f"20260101_{hh:02d}{mm:02d}{ss:02d}_part{i + 1}of{k}.mp4"
            if k == 1:
                shutil.copyfile(master, name)
                continue
            # 元の動画と同じ設定でエンコードし直して切る(-c copy では、B フレームの都合で境目が 1 GOP ずれるため)
            run(["-ss", str(start), "-t", str(part), "-i", str(master), *X264, "-an", str(name)])
        print(f"{out}({k} 本 × {part} 秒)")


def make_quality_set(dest: Path) -> None:
    """品質の評価用の単色の画像(1080p)を、IMAGE_MAX_SIDE の条件ごとの大きさで作ります。"""
    q = dest / "quality"
    for side in QUALITY_SIDES:
        (q / "images" / str(side)).mkdir(parents=True, exist_ok=True)
    for name, color, _ja, _en in QUALITY_COLORS:
        for side in QUALITY_SIDES:
            w, h = side, side * 9 // 16 // 2 * 2
            run(["-f", "lavfi", "-i", f"color=c={color}:size={w}x{h}:duration=1", "-frames:v", "1", "-q:v", "2",
                 str(q / "images" / str(side) / f"{name}.jpg")])  # fmt: skip
    queries = [{"image": name, "ja": ja, "en": en} for name, _c, ja, en in QUALITY_COLORS]
    (q / "queries.json").write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{q}(画像 {len(QUALITY_COLORS)} 枚 × 大きさ {len(QUALITY_SIDES)} 種、文章 {len(queries) * 2} 件)")


def main() -> int:
    ap = argparse.ArgumentParser(description="ベンチマーク用の合成の映像・音声・画像を作ります")
    ap.add_argument("dest")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--images", type=int, default=50)
    ap.add_argument("--split-set", action="store_true", help="共通ベンチマーク(MATRIX.md)の素材だけを作る")
    args = ap.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg が見つかりません")
    dest = Path(args.dest)
    if args.split_set:
        make_split_set(dest)
        make_quality_set(dest)
        return 0
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
