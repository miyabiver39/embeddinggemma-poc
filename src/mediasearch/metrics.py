"""資源の使用状況と処理の状態(GET /api/metrics、GET /compute/metrics)。

ベンチマーク(リポジトリの bench/)が 1 秒ごとに読み、ボトルネックを判断するためのものです。
値は「起動からの積算」(CPU 時間・推論の時間など)と「現在の値」(メモリ・待ち件数など)で、
使用率は、利用側が 2 回読んだ差を経過時間で割って求めます(この API 自身は計算の重い処理をしない)。

- 追加の依存は使わず、Linux の /proc と cgroup(v2 と v1。/sys/fs/cgroup)から読みます。コンテナの中では、
  cgroup の値がコンテナ全体(ffmpeg を含む)、/proc/self がこのプロセスの値です。
- スレッドごとの CPU 時間は、Python のスレッド名(ingestor、decode、stream-1、embed-batch など)でまとめます。
  名前の無いスレッド(PyTorch の演算スレッドなど)は native にまとめます。
- GPU は、nvidia-smi(NVIDIA。コンテナには NVIDIA Container Toolkit が渡す)か、amdgpu の sysfs(AMD)から読みます。
  どちらも無い場合(Intel など)は、PyTorch のメモリの使用量だけを返します。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

_TICK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_CGROUP = Path("/sys/fs/cgroup")
_NVIDIA_SMI = shutil.which("nvidia-smi")


def _read(path: Path | str) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def _stat_cpu(text: str | None) -> float | None:
    """/proc/<pid>/stat の utime + stime(秒)。comm に空白や括弧を含むことがあるため、最後の ')' の後を数える。"""
    if not text:
        return None
    fields = text[text.rfind(")") + 2 :].split()
    return (int(fields[11]) + int(fields[12])) / _TICK  # 14 番目と 15 番目(state から数えて 12・13 番目)


def _thread_cpu() -> dict[str, float]:
    names = {t.native_id: t.name for t in threading.enumerate() if t.native_id is not None}
    out: dict[str, float] = {}
    for task in Path("/proc/self/task").iterdir() if Path("/proc/self/task").is_dir() else []:
        cpu = _stat_cpu(_read(task / "stat"))
        if cpu is None:
            continue
        name = names.get(int(task.name), "native")
        # 番号つきの名前(stream-3、AnyIO worker thread など)は、種類ごとにまとめる
        group = re.sub(r"[-_ ]?\d+$", "", name) or name
        if group.startswith("asyncio"):
            group = "event-loop"
        elif group.startswith("AnyIO worker thread") or group.startswith("ThreadPoolExecutor"):
            group = "worker"  # API の処理(FastAPI のスレッドプール)
        out[group] = round(out.get(group, 0.0) + cpu, 3)
    return out


def _children_cpu() -> dict[str, float]:
    """実行中の子プロセス(ffmpeg など)の CPU 時間を、プログラム名ごとにまとめます。"""
    me = os.getpid()
    out: dict[str, float] = {}
    proc = Path("/proc")
    for entry in proc.iterdir() if proc.is_dir() else []:
        if not entry.name.isdigit():
            continue
        text = _read(entry / "stat")
        if not text:
            continue
        comm = text[text.find("(") + 1 : text.rfind(")")]
        fields = text[text.rfind(")") + 2 :].split()
        if int(fields[1]) != me:  # ppid
            continue
        cpu = (int(fields[11]) + int(fields[12])) / _TICK
        out[comm] = round(out.get(comm, 0.0) + cpu, 3)
    return out


def _kv(text: str | None) -> dict[str, int]:
    out = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            out[parts[0]] = int(parts[1])
    return out


def _cgroup() -> dict:
    """コンテナ全体の CPU・メモリ・ディスクの値。cgroup v2 と v1 の両方に対応し、どちらでもない環境では空。"""
    if (_CGROUP / "cpu.stat").exists() and (_CGROUP / "cgroup.controllers").exists():
        return _cgroup_v2()
    if (_CGROUP / "cpuacct" / "cpuacct.usage").exists():
        return _cgroup_v1()
    return {}


def _cgroup_v2() -> dict:
    cpu = _kv(_read(_CGROUP / "cpu.stat"))
    limit = None
    quota = (_read(_CGROUP / "cpu.max") or "").split()
    if len(quota) == 2 and quota[0] != "max":
        limit = int(quota[0]) / int(quota[1])  # 使える CPU の数(例: 2.0)
    mem_max = (_read(_CGROUP / "memory.max") or "").strip()
    rbytes = wbytes = 0
    for line in (_read(_CGROUP / "io.stat") or "").splitlines():
        for item in line.split()[1:]:
            k, _, v = item.partition("=")
            if k == "rbytes":
                rbytes += int(v)
            elif k == "wbytes":
                wbytes += int(v)
    mem = _kv(_read(_CGROUP / "memory.stat"))
    return {
        "version": 2,
        "cpu_sec": round(cpu.get("usage_usec", 0) / 1e6, 3),
        "throttled_sec": round(cpu.get("throttled_usec", 0) / 1e6, 3),  # CPU の上限で止められた時間
        "cpu_limit": limit,
        "memory_bytes": int((_read(_CGROUP / "memory.current") or "0").strip() or 0),
        "memory_limit_bytes": int(mem_max) if mem_max.isdigit() else None,
        "memory_anon_bytes": mem.get("anon"),  # ページキャッシュを除いた使用量
        "io_read_bytes": rbytes,
        "io_write_bytes": wbytes,
    }


def _cgroup_v1() -> dict:
    """cgroup v1(古い Docker・Linux)。値の場所と単位が v2 と違う。"""

    def num(path: str) -> int | None:
        text = (_read(_CGROUP / path) or "").strip()
        return int(text) if text.lstrip("-").isdigit() else None

    quota, period = num("cpu/cpu.cfs_quota_us"), num("cpu/cpu.cfs_period_us")
    stat = _kv(_read(_CGROUP / "cpu" / "cpu.stat"))
    mem = _kv(_read(_CGROUP / "memory" / "memory.stat"))
    mem_limit = num("memory/memory.limit_in_bytes")
    rbytes = wbytes = 0
    for line in (_read(_CGROUP / "blkio" / "blkio.throttle.io_service_bytes") or "").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] == "Read":
            rbytes += int(parts[2])
        elif len(parts) == 3 and parts[1] == "Write":
            wbytes += int(parts[2])
    return {
        "version": 1,
        "cpu_sec": round((num("cpuacct/cpuacct.usage") or 0) / 1e9, 3),
        "throttled_sec": round(stat.get("throttled_time", 0) / 1e9, 3),
        "cpu_limit": quota / period if quota and quota > 0 and period else None,
        "memory_bytes": num("memory/memory.usage_in_bytes"),
        # 上限なしは非常に大きな値になる(2^63 付近)
        "memory_limit_bytes": mem_limit if mem_limit and mem_limit < 2**60 else None,
        "memory_anon_bytes": mem.get("total_rss", mem.get("rss")),
        "io_read_bytes": rbytes,
        "io_write_bytes": wbytes,
    }


def _host_cpu() -> dict:
    """マシン全体の CPU 時間(/proc/stat。コンテナの中でもホスト全体の値)。"""
    first = (_read("/proc/stat") or "").split("\n", 1)[0].split()
    if not first or first[0] != "cpu":
        return {}
    vals = [int(v) for v in first[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return {"total_sec": round(sum(vals) / _TICK, 2), "idle_sec": round(idle / _TICK, 2)}


def _gpu() -> dict:
    out: dict = {}
    if _NVIDIA_SMI:
        try:
            res = subprocess.run(
                [_NVIDIA_SMI, "--query-gpu=index,name,utilization.gpu,utilization.memory,utilization.decoder,"
                 "memory.used,memory.total,power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )  # fmt: skip
            gpus = []
            for line in res.stdout.strip().splitlines():
                v = [x.strip() for x in line.split(",")]
                if len(v) < 8:
                    continue

                def num(x: str) -> float | None:
                    try:
                        return float(x)
                    except ValueError:
                        return None

                gpus.append({
                    "index": int(v[0]), "name": v[1], "util_pct": num(v[2]), "mem_util_pct": num(v[3]),
                    "decoder_pct": num(v[4]), "memory_used_mb": num(v[5]), "memory_total_mb": num(v[6]),
                    "power_w": num(v[7]),
                })  # fmt: skip
            if gpus:
                out["source"] = "nvidia-smi"
                out["devices"] = gpus
        except (OSError, subprocess.TimeoutExpired):
            pass
    if "devices" not in out:
        gpus = []
        for card in sorted(Path("/sys/class/drm").glob("card[0-9]")) if Path("/sys/class/drm").is_dir() else []:
            dev = card / "device"
            busy = _read(dev / "gpu_busy_percent")  # amdgpu のみ
            if busy is None:
                continue
            used, total = _read(dev / "mem_info_vram_used"), _read(dev / "mem_info_vram_total")
            gpus.append({
                "index": len(gpus), "name": card.name, "util_pct": float(busy.strip()),
                "memory_used_mb": int(used) / 2**20 if used else None,
                "memory_total_mb": int(total) / 2**20 if total else None,
            })  # fmt: skip
        if gpus:
            out["source"] = "amdgpu-sysfs"
            out["devices"] = gpus
    torch = sys.modules.get("torch")  # 推論で読み込み済みのときだけ(ここで読み込むと重いため)
    if torch is not None:
        try:
            if torch.cuda.is_available():
                out["torch"] = {
                    "allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
                    "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                    "reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1),
                }
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                out["torch"] = {"allocated_mb": round(torch.xpu.memory_allocated() / 2**20, 1)}
        except Exception:  # 計測の失敗で API を失敗させない
            pass
    return out


def process_metrics() -> dict:
    """このプロセスとコンテナの資源の使用状況。"""
    t = os.times()
    status: dict[str, int] = {}
    for line in (_read("/proc/self/status") or "").splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM", "Threads") and value.split():
            status[key] = int(value.split()[0])  # VmRSS などは kB
    io = _kv((_read("/proc/self/io") or "").replace(":", ""))
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    return {
        "time": time.time(),
        "monotonic": time.monotonic(),
        "cpus": cpus,
        "process": {
            "cpu_sec": round(t.user + t.system, 3),
            "threads": status.get("Threads"),
            "rss_bytes": status.get("VmRSS", 0) * 1024,
            "rss_peak_bytes": status.get("VmHWM", 0) * 1024,
            "io_read_bytes": io.get("read_bytes"),
            "io_write_bytes": io.get("write_bytes"),
        },
        "threads_cpu_sec": _thread_cpu(),
        # 実行中の子プロセス(ffmpeg など)と、終了した子プロセスの合計
        "children_cpu_sec": {**_children_cpu(), "(終了したもの)": round(t.children_user + t.children_system, 3)},
        "cgroup": _cgroup(),
        "host": _host_cpu(),
        "gpu": _gpu(),
    }


def app_metrics(ctx) -> dict:
    """アプリの処理の状態(推論の使用率・待ち件数・ストリームの遅れなど)。"""
    out = process_metrics()
    out["inference"] = ctx.embedder.metrics()
    store = ctx.store
    out["index"] = {
        "windows": store.last_window_id,
        "vectors": len(store.vectors),
        "journal_pending": store.vectors.pending,
    }
    out["ingest"] = {"queue": ctx.ingestor.queue_size}
    if ctx.streams is not None:
        items = ctx.streams.list()
        out["streams"] = [
            {"id": s["id"], "name": s["name"], **{k: s["state"].get(k) for k in
             ("status", "frames", "windows", "dropped_windows", "lag_ms", "decoder")}}
            for s in items
        ]  # fmt: skip
    return out
