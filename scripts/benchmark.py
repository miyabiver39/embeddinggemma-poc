"""取り込みの処理時間を測ります(高速化の効果を確かめるためのもの)。

起動中のサーバに、同じファイルを指定の回数だけ取り込み直させ、ジョブに記録された処理時間の内訳(timings)を表にします。
Python の標準ライブラリだけで動くため、サーバと別のマシンからでも実行できます。

使い方:
    # サーバ内のパス(INGEST_ROOTS の下)を取り込む
    python scripts/benchmark.py --path /recordings/sample.mp4 --repeat 3
    # 手元のファイルをアップロードして取り込む
    python scripts/benchmark.py --upload ./sample.mp4 --audio
    # 設定を変えて比べる例(サーバ側で FFMPEG_HWACCEL=none にして起動し直してから、もう一度実行する)

オプション:
    --url     サーバの URL(既定 http://localhost:8000)
    --token   API_TOKEN を設定している場合のトークン(環境変数 MEDIASEARCH_TOKEN でも可)
    --preset  取り込みのプリセット(object / action / speech)
    --audio   動画の音声も含める(include_audio)
    --json    結果を JSON で出力する

取り込み直すたびに、そのファイルの以前の窓は削除されます(検索の対象から一時的に外れます)。
本番のデータではなく、確認用のサーバで実行してください。
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

STAGES = ("probe", "decode", "audio", "image", "embed", "store", "thumb")


class Client:
    def __init__(self, url: str, token: str | None) -> None:
        self.url = url.rstrip("/")
        self.token = token

    def request(self, method: str, path: str, body: bytes | None = None, content_type: str | None = None) -> dict:
        req = urllib.request.Request(self.url + path, data=body, method=method)
        if content_type:
            req.add_header("Content-Type", content_type)
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=600) as res:
                return json.loads(res.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise SystemExit(f"{method} {path} が失敗しました({exc.code}): {detail}") from exc

    def post_json(self, path: str, payload: dict) -> dict:
        return self.request("POST", path, json.dumps(payload).encode(), "application/json")

    def post_file(self, path: str, file: Path, fields: dict[str, str]) -> dict:
        boundary = uuid.uuid4().hex
        parts = []
        for k, v in fields.items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        head = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{file.name}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n"
        )
        body = b"".join(parts) + head.encode() + file.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        return self.request("POST", path, body, f"multipart/form-data; boundary={boundary}")

    def wait(self, job_id: int) -> dict:
        while True:
            job = self.request("GET", f"/api/jobs/{job_id}")
            if job["status"] in ("done", "failed"):
                return job
            time.sleep(0.2)


def run_once(c: Client, args: argparse.Namespace, source_id: int | None) -> tuple[int, dict]:
    """1 回取り込み、(source_id, ジョブ) を返します。2 回目以降は取り込み直し(reindex)を使います。"""
    if source_id is not None:
        job_id = c.request("POST", f"/api/sources/{source_id}/reindex")["job_id"]
        return source_id, c.wait(job_id)
    options = {"force": True}
    if args.preset:
        options["preset"] = args.preset
    if args.audio:
        options["include_audio"] = True
    if args.path:
        accepted = c.post_json("/api/ingest/path", {"path": args.path, **options})
    else:
        file = Path(args.upload)
        kind = {"video": "video", "audio": "audio", "image": "image"}.get(
            (mimetypes.guess_type(file.name)[0] or "video/").split("/")[0], "video"
        )
        fields = {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in options.items() if k != "force"}
        if kind == "image":
            raise SystemExit("画像は --path で指定してください(アップロードの API が複数枚の形式のため)")
        accepted = c.post_file(f"/api/ingest/{kind}", file, fields)
    return accepted["source_id"], c.wait(accepted["job_id"])


def summarize(jobs: list[dict]) -> dict:
    timings = [j["timings"] for j in jobs if j.get("timings")]
    if not timings:
        return {}

    def med(values: list[float]) -> float:
        return round(statistics.median(values), 1)

    stages = {s: med([t["stages_ms"].get(s, 0.0) for t in timings]) for s in STAGES}
    return {
        "runs": len(timings),
        "total_ms": med([t["total_ms"] for t in timings]),
        "stages_ms": {k: v for k, v in stages.items() if v > 0},
        "windows": timings[-1]["windows"],
        "media_ms": timings[-1]["media_ms"],
        "per_window_ms": med([t["per_window_ms"] or 0 for t in timings]),
        "realtime_factor": med([t["realtime_factor"] or 0 for t in timings]),
        "decoder": timings[-1].get("decoder"),
        "accelerator": timings[-1].get("accelerator"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="取り込みの処理時間を測ります")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--path", help="サーバ内のファイルのパス")
    src.add_argument("--upload", help="アップロードする手元のファイル(動画・音声)")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--token", default=os.environ.get("MEDIASEARCH_TOKEN"))
    ap.add_argument("--repeat", type=int, default=1, help="繰り返す回数(中央値を表示)")
    ap.add_argument("--preset", choices=("object", "action", "speech"))
    ap.add_argument("--audio", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    c = Client(args.url, args.token)
    info = c.request("GET", "/api/info")
    jobs, source_id = [], None
    for n in range(1, max(args.repeat, 1) + 1):
        source_id, job = run_once(c, args, source_id)
        if job["status"] != "done":
            print(f"取り込みに失敗しました: {job.get('error')}", file=sys.stderr)
            return 1
        jobs.append(job)
        t = job.get("timings") or {}
        if not args.json:
            rf = t.get("realtime_factor")
            print(f"[{n}/{args.repeat}] {t.get('total_ms')} ms(窓 {t.get('windows')}、実時間比 {rf})")
    result = summarize(jobs)
    result["server"] = {"version": info.get("version"), "embedder": info.get("embedder")}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if not result.get("runs"):
        print("サーバが処理時間を記録していません(古い版の可能性があります)", file=sys.stderr)
        return 1
    print()
    print(
        f"中央値({result['runs']} 回): {result['total_ms']} ms / 窓 {result['windows']}"
        f" / 1 窓 {result['per_window_ms']} ms / 実時間比 {result['realtime_factor']}"
    )
    print(f"復号: {result['decoder']}  推論: {result['accelerator']}")
    for stage, ms in result["stages_ms"].items():
        share = ms / result["total_ms"] * 100 if result["total_ms"] else 0
        print(f"  {stage:<7} {ms:>10.1f} ms  {share:5.1f}%")
    print("(decode は推論側がフレームを待った時間。復号は推論と並行するため、合計は total を超えないことがあります)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
