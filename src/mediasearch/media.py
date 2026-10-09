"""ffmpeg / ffprobe による映像・音声の加工。

モデルに渡す前の加工(フレーム抽出・16kHz モノラル化)は、すべてここで行います。
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .embedders.base import SAMPLE_RATE

log = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------- 連続復号(取り込み用)
def _even(n: float) -> int:
    """yuv 系の形式は幅・高さが偶数でないと変換できないため、偶数に丸める。"""
    return max(2, int(round(n / 2)) * 2)


def output_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """長辺を max_side 以下に縮小した大きさ(縦横比は保つ)。"""
    if max_side <= 0 or max(width, height) <= max_side:
        return _even(width), _even(height)
    scale = max_side / max(width, height)
    return _even(width * scale), _even(height * scale)


def video_size(path: str | Path) -> tuple[int, int]:
    """映像の幅と高さ(回転の指定を反映)。"""
    out = _run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-print_format", "json", str(path)]
    )
    streams = json.loads(out).get("streams") or []
    if not streams:
        raise MediaError("映像トラックがありません")
    s = streams[0]
    width, height = int(s.get("width") or 0), int(s.get("height") or 0)
    rotation = 0
    for side in s.get("side_data_list") or []:
        rotation = int(side.get("rotation") or 0) or rotation
    if abs(rotation) in (90, 270):  # スマートフォンの縦撮りなど。ffmpeg は自動で回転するため、出力の縦横も入れ替える
        width, height = height, width
    if width <= 0 or height <= 0:
        raise MediaError("映像の大きさを取得できません")
    return width, height


_HWACCELS: set[str] | None = None


def available_hwaccels() -> set[str]:
    """この ffmpeg が対応している GPU 復号の方式(ffmpeg -hwaccels の結果。1 回だけ調べる)。"""
    global _HWACCELS
    if _HWACCELS is None:
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True, timeout=30)
            _HWACCELS = {line.strip() for line in out.stdout.splitlines()[1:] if line.strip()}
        except (OSError, subprocess.TimeoutExpired):
            _HWACCELS = set()
    return _HWACCELS


def select_hwaccel(setting: str, device: str = "") -> tuple[str | None, str | None]:
    """FFMPEG_HWACCEL の設定から、使う GPU 復号の方式とデバイスを決めます。使わない場合は (None, None)。

    auto: NVIDIA の GPU がコンテナに渡されていれば cuda、/dev/dri(AMD / Intel)があれば vaapi、どちらもなければ CPU。
    GPU で復号できなかった場合は、VideoFrameReader が CPU でやり直すため、誤って選んでも取り込みは止まらない。
    """
    setting = (setting or "auto").strip().lower()
    if setting in ("none", "cpu", "off", "false", "0"):
        return None, None
    render_nodes = sorted(str(p) for p in Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").is_dir() else []
    if setting == "auto":
        if Path("/dev/nvidiactl").exists() or Path("/dev/nvidia0").exists():
            setting = "cuda"
        elif render_nodes:
            setting = "vaapi"
        else:
            return None, None
    if setting not in available_hwaccels():
        log.warning("ffmpeg が GPU 復号 %s に対応していないため、CPU で復号します", setting)
        return None, None
    if setting == "vaapi" and not device:
        device = render_nodes[0] if render_nodes else ""
    return setting, device or None


# 復号を省く範囲(FFMPEG_SKIP_FRAMES)。ffmpeg のデコーダの skip_frame に渡す
#   none:      全フレームを復号する(最も正確・最も遅い)
#   noref:     他のフレームから参照されないフレーム(B フレームなど)を復号しない。
#              取り出し時刻のずれは 1〜2 フレーム程度で、1080p の H.264 で約 25% 速い(既定)。
#              参照されないフレームが無い形式では、none と同じ
#   keyframes: キーフレームだけを復号する。数倍速いが、取り出し時刻が最大でキーフレームの間隔(数秒のことが多い)ずれる
DECODE_SKIP: dict[str, list[str]] = {
    "none": [],
    "noref": ["-skip_frame", "noref"],
    "keyframes": ["-skip_frame", "nokey"],
}


class VideoFrameReader:
    """1 つの ffmpeg プロセスで動画を先頭から復号し、一定間隔のフレームを順に返します。

    フレームごとに ffmpeg を起動して頭出しする方式では、起動のたびにキーフレームから復号し直すため、
    1 枚あたり数十〜数百ミリ秒かかる。ここでは 1 回の起動で最後まで流し、ffmpeg の fps フィルタで
    一定間隔のフレームだけを縮小して受け取る(JPEG への変換もしない)。

    hwaccel に cuda / vaapi などを渡すと、復号を GPU で行う(縮小は CPU)。GPU で始められなかった場合は、
    ソフトウェアの復号でやり直す(fallback が真のとき)。
    """

    def __init__(
        self,
        path: str | Path,
        step_ms: int,
        max_side: int = 448,
        hwaccel: str | None = None,
        hwaccel_device: str | None = None,
        skip: str = "noref",
    ) -> None:
        if skip not in DECODE_SKIP:
            raise ValueError(f"復号の省略の指定は {list(DECODE_SKIP)} のいずれかにしてください: {skip!r}")
        self.path = str(path)
        self.skip = skip
        self.skip_used = skip
        self.step_ms = max(int(step_ms), 1)
        self.width, self.height = output_size(*video_size(path), max_side)
        self.hwaccel = hwaccel
        self.hwaccel_device = hwaccel_device
        self.decoder = "cpu"
        self.frames = 0

    def _command(self, hwaccel: str | None, skip: str) -> list[str]:
        cmd = ["ffmpeg", "-v", "error", "-nostdin", *DECODE_SKIP[skip]]
        if hwaccel:
            cmd += ["-hwaccel", hwaccel]
            if self.hwaccel_device:
                cmd += ["-hwaccel_device", self.hwaccel_device]
        fps = 1000.0 / self.step_ms
        cmd += [
            "-i", self.path, "-an", "-sn", "-dn",
            # round=near: 各出力時刻に最も近い元のフレームを使う。area: 縮小のときに画質がよい
            "-vf", f"fps=fps={fps:.6f}:round=near,scale={self.width}:{self.height}:flags=area",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
        ]  # fmt: skip
        return cmd

    def _attempts(self) -> list[tuple[str | None, str]]:
        """試す順の (GPU 復号, 省略の範囲)。GPU で始められない場合は CPU で、省略して 1 枚も取れない場合
        (キーフレームが先頭の 1 枚だけの短い動画など)は全フレームの復号でやり直す。"""
        out: list[tuple[str | None, str]] = []
        for item in ((self.hwaccel, self.skip), (None, self.skip), (None, "none")):
            if item not in out:
                out.append(item)
        return out

    def __iter__(self):
        """(フレームの時刻 ms, RGB 配列) を順に返します。"""
        err = ""
        for hw, skip in self._attempts():
            yielded = 0
            proc = subprocess.Popen(self._command(hw, skip), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            size = self.width * self.height * 3
            try:
                while True:
                    buf = proc.stdout.read(size)
                    if len(buf) < size:
                        break
                    frame = np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3)
                    self.decoder = hw or "cpu"
                    self.skip_used = skip
                    yield yielded * self.step_ms, frame
                    yielded += 1
                    self.frames += 1
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
                err = proc.stderr.read().decode("utf-8", "replace").strip() if proc.stderr else ""
            if yielded > 0:
                return
            log.warning(
                "復号(GPU=%s, 省略=%s)でフレームを取り出せないため、条件を変えてやり直します: %s",
                hw or "なし", skip, err.splitlines()[-1:] or "",
            )  # fmt: skip
        reason = " / ".join(err.splitlines()[-3:]) or "フレームがありません"
        raise MediaError(f"ffmpeg で映像を復号できません: {reason}")


class AudioTrack:
    """動画の音声を 1 回の ffmpeg で 16kHz モノラルに変換し、区間ごとに取り出せるようにします。

    窓ごとに ffmpeg を起動して頭出しする代わりに、全体を一時ファイルに書き出してメモリマップで読む
    (長い動画でもメモリを圧迫しない)。変換は別プロセスで、映像の復号と並行して進める。
    """

    def __init__(self, path: str | Path, tmp_dir: str | Path) -> None:
        self._file = Path(tmp_dir) / "audio.s16le"
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(path), "-vn", "-ac", "1",
               "-ar", str(SAMPLE_RATE), "-f", "s16le", str(self._file)]  # fmt: skip
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self._data: np.ndarray | None = None

    def _ready(self) -> np.ndarray:
        if self._data is None:
            _, err = self._proc.communicate(timeout=3600)
            if self._proc.returncode != 0:
                raise MediaError(f"ffmpeg で音声を変換できません: {err.decode('utf-8', 'replace').strip()[-300:]}")
            if self._file.stat().st_size == 0:
                self._data = np.zeros(0, dtype="<i2")
            else:
                self._data = np.memmap(self._file, dtype="<i2", mode="r")
        return self._data

    def slice(self, start_ms: int, end_ms: int) -> np.ndarray:
        data = self._ready()
        lo = max(start_ms, 0) * SAMPLE_RATE // 1000
        hi = max(end_ms, start_ms) * SAMPLE_RATE // 1000
        return np.asarray(data[lo:hi], dtype=np.float32) / 32768.0

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()
        self._data = None


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


# ---------------------------------------------------------------------- 静止画
def load_image(path: str | Path, max_side: int = 448) -> np.ndarray:
    """静止画を RGB 配列で読み込みます。EXIF の向き(スマートフォンの縦撮りなど)を反映し、長辺を max_side に縮小します。

    動画から切り出せない場合でも、画像だけで登録できるようにするための入口です。
    """
    from PIL import ImageOps, UnidentifiedImageError

    try:
        with Image.open(path) as img:
            if max_side and img.format == "JPEG":
                # JPEG は縮小しながら復号できる(1/2〜1/8)。大きな写真ほど読み込みが速くなる
                img.draft("RGB", (max_side * 2, max_side * 2))
            img = ImageOps.exif_transpose(img).convert("RGB")
            if max_side and max(img.size) > max_side:
                img.thumbnail((max_side, max_side))
            return np.asarray(img)
    except (UnidentifiedImageError, OSError) as exc:
        raise MediaError(f"画像として読めません: {Path(path).name}({exc})") from exc


def check_image(path: str | Path) -> tuple[int, int]:
    """画像として読めるか確かめ、(幅, 高さ) を返します。全体を復号せずに確認するため速く済みます。"""
    from PIL import UnidentifiedImageError

    try:
        with Image.open(path) as img:
            size = img.size  # verify の後は画像を使えないため、先に大きさを取っておく
            img.verify()  # ファイルの破損を検出する
            return size
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise MediaError(f"画像として読めません: {Path(path).name}({exc})") from exc


def image_taken_at(path: str | Path) -> float | None:
    """EXIF の撮影日時(DateTimeOriginal、なければ DateTime)を UNIX 秒で返します。無ければ None。

    EXIF の日時にはタイムゾーンが無いため、コンテナの設定(TZ)として解釈します。
    """
    from datetime import datetime

    try:
        with Image.open(path) as img:
            exif = img.getexif()
            # 0x8769: 詳細情報(Exif IFD)。DateTimeOriginal(0x9003)はこの中にある。0x0132: DateTime
            value = exif.get_ifd(0x8769).get(0x9003) or exif.get(0x0132)
    except Exception:  # EXIF の破損は、撮影日時が無いものとして扱う
        return None
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip()[:19], "%Y:%m:%d %H:%M:%S").timestamp()
    except ValueError:
        return None
