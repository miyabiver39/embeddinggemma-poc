"""共通ベンチマーク(同時取り込み × 分割 × 窓 × 精度)を、決まった手順で実行します。

決まり(条件・手順・記録の形式)は bench/MATRIX.md にあります。この文書と、このスクリプトで、機種をまたいで同じ計測をします。
条件ごとにコンテナを起動し直し(DTYPE / IMAGE_MAX_SIDE / DIMS / INGEST_WORKERS)、取り込み・検索・品質を測って、
results.jsonl に 1 回ずつ追記します。途中で止まっても --resume で続きから再開できます。
Python の標準ライブラリと docker の CLI だけで動きます。

使い方の例:
    python bench/matrix.py --machine rtx3060 --image ghcr.io/miyabiver39/embeddinggemma-poc:cuda \\
        --docker-args "--gpus all" --media ./bench-media --out bench/results/matrix-rtx3060
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze import derive, summarize_rows  # noqa: E402
from sampler import ApiError, Client, Sampler  # noqa: E402

from bench import DEFAULT_QUERIES  # noqa: E402

TOTAL_SEC = 480  # 素材の合計の長さ(make_samples.py --split-set)
DEFAULT_PRECISION = ("bfloat16", 448, 768)
BASELINE_PRECISION = ("float32", 448, 768)  # 品質の基準


# ---------------------------------------------------------------------- 計測の組み合わせ
def plan(quick: bool, experiments: set[str], reps: int) -> list[dict]:
    """起動するコンテナの順番と、それぞれで行う計測を返します。

    1 つの要素が 1 回の起動です。同じ条件の計測は、まとめて同じコンテナで行います(起動の回数を減らすため)。
    """
    containers: list[dict] = []
    if quick:
        ks, ws, wins, reps = (1, 2), (1, 2), (2,), 1
        precisions = [DEFAULT_PRECISION]
    else:
        ks, ws, wins = (1, 2, 4, 8), (1, 2, 4, 8), (1, 2, 4, 6)
        precisions = [(d, s, m) for d in ("bfloat16", "float16", "float32") for s in (448, 336, 224)
                      for m in (768, 256, 128)]  # fmt: skip
    if "concurrency" in experiments:
        for w in ws:
            cells = [{"experiment": "concurrency", "k": k, "w": w, "win": win} for win in wins for k in ks if k >= w]
            containers.append({"precision": DEFAULT_PRECISION, "w": w, "cells": cells, "reps": reps,
                               "search": False, "quality": False})  # fmt: skip
    if "precision" in experiments:
        # 基準(float32・448・768)を先に測り、品質の比較に使う
        order = sorted(precisions, key=lambda p: p != BASELINE_PRECISION)
        pk = max(ks)
        for prec in order:
            for w in (1, 4) if not quick else (1,):
                cells = [{"experiment": "precision", "k": pk, "w": w, "win": 2}]
                containers.append({"precision": prec, "w": w, "cells": cells, "reps": reps,
                                   "search": w == 1, "quality": w == 1})  # fmt: skip
    return containers


def cond_key(rec: dict) -> tuple:
    return (rec["experiment"], rec["dtype"], rec["image_max_side"], rec["dims"], rec.get("k"), rec.get("w"),
            rec.get("win"), rec.get("rep"))  # fmt: skip


# ---------------------------------------------------------------------- コンテナ
class Server:
    """計測用のコンテナを、条件を変えて起動し直します。"""

    def __init__(self, args) -> None:
        self.args = args
        self.name = args.container_name
        self.url = f"http://127.0.0.1:{args.port}"
        self.data_root = Path(args.data_root).resolve()
        self.media = Path(args.media).resolve()

    def stop(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)

    def start(self, prec: tuple, w: int) -> Client:
        dtype, side, dims = prec
        self.stop()
        data = self.data_root / f"data-d{dims}"  # DB は次元を記録して照合するため、DIMS ごとに分ける
        cache = self.data_root / "torch-cache"  # compile の結果は共通にして、作り直しを避ける
        for d in (data, cache):
            d.mkdir(parents=True, exist_ok=True)
            os.chmod(d, 0o777)  # コンテナの利用者(uid 1000)が書けるように
        env = {"DTYPE": dtype, "IMAGE_MAX_SIDE": side, "DIMS": dims, "INGEST_WORKERS": w,
               "TORCH_CACHE_DIR": "/torch-cache", "LOG_LEVEL": "WARNING"}  # fmt: skip
        cmd = ["docker", "run", "-d", "--name", self.name, "-p", f"127.0.0.1:{self.args.port}:8000",
               "--stop-timeout", "30", "-v", f"{data}:/data", "-v", f"{cache}:/torch-cache",
               "-v", f"{self.media}:/recordings:ro"]  # fmt: skip
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += shlex.split(self.args.docker_args or "") + [self.args.image]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"コンテナを起動できません: {res.stderr.strip()}")
        client = Client(self.url, None, timeout=600)
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.args.start_timeout:
            state = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", self.name],
                                   capture_output=True, text=True).stdout.strip()  # fmt: skip
            if state != "true":
                raise RuntimeError("コンテナが終了しました:\n" + self.logs())
            try:
                client.get("/healthz")
                info = client.get("/api/info")
                print(f"  起動しました({time.monotonic() - t0:.0f} 秒)。推論 {info.get('embedder')}")
                return client
            except (ApiError, OSError):  # 起動の途中は、接続を切られることがある
                time.sleep(2)
        raise RuntimeError(f"{self.args.start_timeout} 秒たっても /healthz が応答しません:\n" + self.logs())

    def logs(self, tail: int = 40) -> str:
        res = subprocess.run(["docker", "logs", "--tail", str(tail), self.name], capture_output=True, text=True)
        return (res.stdout + res.stderr).strip()


# ---------------------------------------------------------------------- 計測
def files_for(media: Path, k: int) -> list[str]:
    names = sorted(p.name for p in (media / "split" / f"k{k}").glob("*.mp4"))
    if len(names) != k:
        raise SystemExit(f"素材が足りません: {media / 'split' / f'k{k}'}(make_samples.py --split-set を実行してください)")
    return [f"/recordings/split/k{k}/{n}" for n in names]


def delete_sources(c: Client, ids: list[int]) -> None:
    for sid in ids:
        try:
            c.delete(f"/api/sources/{sid}")
        except ApiError as exc:
            print(f"  (取り込み元 {sid} を削除できませんでした: {exc})")


def ingest_once(c: Client, paths: list[str], win: int, video_sec: float = TOTAL_SEC) -> dict:
    """paths を一度に投入し、全部が終わるまで待ちます。完了時間はサーバー側のジョブの時刻で測ります。"""
    opts = {"window_sec": win, "frames_per_window": win, "overlap_sec": 0, "include_audio": False, "force": True}
    sampler = Sampler(c, None, 1.0)
    sampler.start()
    try:
        accepted = [c.post("/api/ingest/path", {"path": p, **opts}) for p in paths]
        jobs = [c.wait_job(a["job_id"], poll=0.2) for a in accepted]
    finally:
        sampler.stop()
    summary = summarize_rows(derive(sampler.samples))
    failed = [j for j in jobs if j["status"] != "done"]
    t0 = min(j["created_at"] for j in jobs)
    t1 = max(j.get("finished_at") or 0 for j in jobs)
    windows = sum((j.get("timings") or {}).get("windows") or j.get("total") or 0 for j in jobs)
    completion = t1 - t0

    def metric(key: str, field: str = "avg"):
        m = summary.get(key)
        return m[field] if m else None

    return {
        "ok": not failed,
        "error": failed[0].get("error") if failed else None,
        "completion_sec": round(completion, 3),
        "windows": windows,
        "sec_per_window": round(completion / windows, 4) if windows else None,
        "realtime_factor": round(video_sec / completion, 3) if completion else None,
        "infer_util": metric("inference_busy_pct"),
        "gpu_util": metric("gpu_util_pct"),
        "gpu_mem_max_mb": metric("gpu_mem_mb", "max"),
        "cpu_util": metric("cpu_pct"),
        "cpu_by_thread": summary.get("cpu_by_thread"),
        "jobs": [{k: j.get(k) for k in ("id", "status", "created_at", "started_at", "finished_at", "timings")}
                 for j in jobs],  # fmt: skip
        "_source_ids": [a["source_id"] for a in accepted],
    }


def search_once(c: Client, seconds: float, concurrency: int = 4) -> dict:
    """bench.py search と同じ方法(組み込みの 10 件の文章を、並行 concurrency で繰り返す)で測ります。"""
    lat: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()
    deadline = time.monotonic() + seconds

    def worker(i: int) -> None:
        n = i
        while time.monotonic() < deadline:
            q = DEFAULT_QUERIES[n % len(DEFAULT_QUERIES)]
            n += concurrency
            t = time.perf_counter()
            try:
                c.post("/api/search/text", {"query": q, "top_k": 10})
            except ApiError as exc:
                with lock:
                    errors.append(str(exc))
                continue
            with lock:
                lat.append((time.perf_counter() - t) * 1000)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(concurrency)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    s = sorted(lat)

    def pct(p: float):
        return round(s[min(len(s) - 1, int(len(s) * p))], 1) if s else None

    return {"ok": bool(lat) and not errors, "error": errors[0] if errors else None, "requests": len(lat),
            "qps": round(len(lat) / seconds, 2), "concurrency": concurrency,
            "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99)}}  # fmt: skip


def _post_images(c: Client, files: list[Path], dims: int) -> list[list[float]]:
    boundary = "matrixbench" + base64.b16encode(os.urandom(6)).decode()
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="dims"\r\n\r\n{dims}\r\n'.encode()]
    for f in files:
        head = (f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{f.name}"\r\n'
                "Content-Type: image/jpeg\r\n\r\n")  # fmt: skip
        parts.append(head.encode() + f.read_bytes() + b"\r\n")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    return c.request("POST", "/compute/embed/images", body, f"multipart/form-data; boundary={boundary}")["vectors"]


def quality_once(c: Client, media: Path, side: int, dims: int) -> tuple[dict, dict]:
    """評価用の画像と文章をベクトルにし、文章から画像を探した 1 位の正解率と点数の差を返します。

    基準との cos 類似度は、全条件のベクトルがそろってから matrix_report.py で計算します(ベクトルは別のファイルに保存)。
    """
    q = media / "quality"
    queries = json.loads((q / "queries.json").read_text(encoding="utf-8"))
    names = [x["image"] for x in queries]
    img_vecs = _post_images(c, [q / "images" / str(side) / f"{n}.jpg" for n in names], dims)
    texts = [x["ja"] for x in queries] + [x["en"] for x in queries]
    txt_vecs = c.post("/compute/embed/texts", {"texts": texts, "kind": "query", "dims": dims})["vectors"]
    nan = any(math.isnan(v) for vec in img_vecs + txt_vecs for v in vec)
    correct, margins = 0, []
    for i, tv in enumerate(txt_vecs):
        answer = i % len(names)
        scores = [sum(a * b for a, b in zip(tv, iv, strict=True)) for iv in img_vecs]
        best = max(range(len(scores)), key=lambda j: scores[j])
        correct += best == answer
        others = max(s for j, s in enumerate(scores) if j != answer)
        margins.append(scores[answer] - others)
    rec = {"ok": not nan, "error": "NaN を含むベクトルがあります" if nan else None, "nan": nan,
           "top1_acc": round(correct / len(txt_vecs), 4), "margin": round(statistics.fmean(margins), 5),
           "queries": len(txt_vecs)}  # fmt: skip
    return rec, {"images": img_vecs, "texts": txt_vecs, "names": names}


# ---------------------------------------------------------------------- 実行
def machine_info(args, image_digest: str, revision: str) -> dict:
    def sh(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return ""

    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem_gb = None
    try:
        kb = int(Path("/proc/meminfo").read_text().split()[1])
        mem_gb = round(kb / 1024 / 1024, 1)
    except (OSError, ValueError, IndexError):
        pass
    gpu = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    if not gpu and shutil.which("rocm-smi"):
        gpu = sh(["rocm-smi", "--showproductname"])
    return {
        "machine": args.machine,
        "gpu": gpu or None,
        "cpu": cpu or platform.processor(),
        "cpu_count": os.cpu_count(),
        "mem_gb": mem_gb,
        "os": platform.platform(),
        "docker": sh(["docker", "version", "--format", "{{.Server.Version}}"]),
        "image": args.image,
        "image_revision": revision,
        "image_digest": image_digest,
        "docker_args": args.docker_args,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "notes": args.notes or "",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="共通ベンチマーク(bench/MATRIX.md)を実行します")
    ap.add_argument("--machine", required=True, help="機種の名前(例 rtx3060、rx9060xt)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--docker-args", default="", help='GPU を渡す引数(例 "--gpus all")')
    ap.add_argument("--media", required=True, help="make_samples.py --split-set で作ったフォルダ")
    ap.add_argument("--out", required=True, help="結果の保存先(results.jsonl などを置く)")
    ap.add_argument("--data-root", help="コンテナの /data と compile のキャッシュを置く場所(既定 <out>/server)")
    ap.add_argument("--experiments", default="concurrency,precision", help="concurrency / precision(カンマ区切り)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--search-sec", type=float, default=15)
    ap.add_argument("--quick", action="store_true", help="動作の確認用(k=1,2・w=1,2・窓 2 秒・1 回)")
    ap.add_argument("--resume", action="store_true", help="results.jsonl にある成功した計測を飛ばして続ける")
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--container-name", default="mediasearch-matrix")
    ap.add_argument("--start-timeout", type=float, default=900)
    ap.add_argument("--notes", help="決まりから外れた点・気づいたこと(machine.json に記録)")
    args = ap.parse_args(argv)
    if shutil.which("docker") is None:
        raise SystemExit("docker が見つかりません")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    args.data_root = args.data_root or str(out / "server")
    media = Path(args.media).resolve()
    results = out / "results.jsonl"
    if results.exists() and not args.resume:
        raise SystemExit(f"{results} があります。続きから再開するなら --resume を付けてください(別の計測なら --out を変える)")
    done: set[tuple] = set()
    if results.exists():
        for line in results.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("ok"):
                done.add(cond_key(rec))

    insp = subprocess.run(["docker", "image", "inspect", args.image, "--format",
                           '{{index .Config.Labels "org.opencontainers.image.revision"}}|{{index .RepoDigests 0}}'],
                          capture_output=True, text=True)  # fmt: skip
    if insp.returncode != 0:
        raise SystemExit(f"イメージがありません。先に docker pull {args.image} を実行してください")
    revision, _, digest = insp.stdout.strip().partition("|")
    info = machine_info(args, digest, revision)
    mfile = out / "machine.json"
    if not mfile.exists() or not args.resume:
        mfile.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"機種 {args.machine}、イメージ {args.image}(revision {revision[:7]})")

    experiments = {e.strip() for e in args.experiments.split(",") if e.strip()}
    containers = plan(args.quick, experiments, args.reps)
    server = Server(args)
    qdir = out / "quality"
    qdir.mkdir(exist_ok=True)

    def write(rec: dict) -> None:
        rec = {k: v for k, v in rec.items() if not k.startswith("_")}
        rec["machine"] = args.machine
        rec["at"] = datetime.now().isoformat(timespec="seconds")
        with results.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    try:
        for n, cont in enumerate(containers, 1):
            dtype, side, dims = cont["precision"]
            base = {"dtype": dtype, "image_max_side": side, "dims": dims}
            todo = [(cell, rep) for rep in range(1, cont["reps"] + 1) for cell in cont["cells"]
                    if cond_key({**base, **cell, "rep": rep}) not in done]  # fmt: skip
            need_search = cont["search"] and cond_key({**base, "experiment": "search", "w": cont["w"]}) not in done
            need_quality = cont["quality"] and cond_key({**base, "experiment": "quality"}) not in done
            if not todo and not need_search and not need_quality:
                continue
            print(f"[{n}/{len(containers)}] {dtype}・{side}・{dims}次元・INGEST_WORKERS={cont['w']}"
                  f"(計測 {len(todo)} 回)")  # fmt: skip
            try:
                c = server.start(cont["precision"], cont["w"])
            except RuntimeError as exc:
                print(f"  起動に失敗しました: {exc}")
                for cell, rep in todo:
                    write({**base, **cell, "rep": rep, "ok": False, "error": f"起動に失敗: {str(exc)[:2000]}"})
                continue
            if need_quality:
                try:
                    rec, vecs = quality_once(c, media, side, dims)
                except ApiError as exc:
                    rec, vecs = {"ok": False, "error": str(exc)}, None
                if vecs:
                    (qdir / f"{dtype}-{side}-{dims}.json").write_text(json.dumps(vecs), encoding="utf-8")
                write({**base, "experiment": "quality", **rec})
                print(f"  品質: 1 位の正解率 {rec.get('top1_acc')}、NaN {rec.get('nan')}")
            leftover: list[int] = []
            for win in sorted({cell["win"] for cell, _ in todo}):
                # ウォームアップ(窓の枚数ごとに compile などの準備が入るため。記録するが集計には使わない)
                try:
                    r = ingest_once(c, files_for(media, 8)[:1], win, TOTAL_SEC / 8)
                    delete_sources(c, r["_source_ids"])
                    write({**base, "experiment": todo[0][0]["experiment"], "k": 0, "w": cont["w"], "win": win,
                           "rep": 0, **r})  # fmt: skip
                except ApiError as exc:
                    print(f"  ウォームアップに失敗しました: {exc}")
                for cell, rep in [(cl, rp) for cl, rp in todo if cl["win"] == win]:
                    delete_sources(c, leftover)
                    try:
                        r = ingest_once(c, files_for(media, cell["k"]), win)
                    except ApiError as exc:
                        r = {"ok": False, "error": str(exc), "_source_ids": []}
                    leftover = r.get("_source_ids") or []
                    write({**base, **cell, "rep": rep, **r})
                    print(f"  k={cell['k']} w={cell['w']} 窓 {win} 秒 #{rep}: "
                          f"{r.get('completion_sec')} 秒(窓 {r.get('windows')}、推論の稼働率 {r.get('infer_util')}%)"
                          + ("" if r["ok"] else f" 失敗: {r.get('error')}"))  # fmt: skip
            if need_search:
                if not leftover:  # 検索の対象が無ければ、8 本を取り込んでから測る
                    r = ingest_once(c, files_for(media, 8), 2)
                    leftover = r["_source_ids"]
                rec = search_once(c, args.search_sec)
                write({**base, "experiment": "search", "w": cont["w"], **rec})
                print(f"  検索(並行 4): {rec['qps']} 件/秒、p50 {rec['latency_ms']['p50']} ms")
            delete_sources(c, leftover)
    finally:
        server.stop()
    print(f"\n完了しました: {results}")
    print(f"集計: python bench/matrix_report.py {out} --out bench/matrix-results/{args.machine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
