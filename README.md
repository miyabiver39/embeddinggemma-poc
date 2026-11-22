# mediasearch — Cross-search video, audio, images and text with EmbeddingGemma 2 (for development)

English | [日本語](README.ja.md)

[![build](https://github.com/miyabiver39/embeddinggemma-poc/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/miyabiver39/embeddinggemma-poc/actions/workflows/build.yml)
[![tests](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmiyabiver39%2Fembeddinggemma-poc%2Fbadges%2Ftests.json)](https://github.com/miyabiver39/embeddinggemma-poc/actions/workflows/build.yml)
[![coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmiyabiver39%2Fembeddinggemma-poc%2Fbadges%2Fcoverage.json)](https://github.com/miyabiver39/embeddinggemma-poc/actions/workflows/build.yml)
[![security scan](https://github.com/miyabiver39/embeddinggemma-poc/actions/workflows/security.yml/badge.svg?branch=main)](https://github.com/miyabiver39/embeddinggemma-poc/actions/workflows/security.yml)
[![vulnerabilities](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmiyabiver39%2Fembeddinggemma-poc%2Fbadges%2Fvulnerabilities.json)](docs/release.md)
[![CodeQL](https://github.com/miyabiver39/embeddinggemma-poc/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/miyabiver39/embeddinggemma-poc/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/miyabiver39/embeddinggemma-poc/badge)](https://scorecard.dev/viewer/?uri=github.com/miyabiver39/embeddinggemma-poc)

[![License](https://img.shields.io/github/license/miyabiver39/embeddinggemma-poc)](LICENSE)
[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fmiyabiver39%2Fembeddinggemma-poc%2Fmain%2Fpyproject.toml)](pyproject.toml)
[![Release](https://img.shields.io/github/v/release/miyabiver39/embeddinggemma-poc?include_prereleases&sort=semver)](https://github.com/miyabiver39/embeddinggemma-poc/releases)
[![Container](https://img.shields.io/badge/container-ghcr.io-blue?logo=docker&logoColor=white)](https://github.com/miyabiver39/embeddinggemma-poc/pkgs/container/embeddinggemma-poc)
[![SBOM](https://img.shields.io/badge/SBOM-SPDX%20%7C%20CycloneDX-blue)](docs/release.md)
[![Dependabot](https://img.shields.io/badge/Dependabot-enabled-brightgreen?logo=dependabot)](.github/dependabot.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Platform](https://img.shields.io/badge/platform-linux%2Famd64-lightgrey)](docker/Dockerfile)
[![GPU](https://img.shields.io/badge/GPU-CUDA%20%7C%20ROCm%20%7C%20Intel%20XPU-informational)](docs/gpu.md)
[![Last commit](https://img.shields.io/github/last-commit/miyabiver39/embeddinggemma-poc)](https://github.com/miyabiver39/embeddinggemma-poc/commits/main)

A backend for finding the **matching moments** in recorded video with **words (or an image or a sound)**, such as
"a scene with a red car" or "the part where a siren is sounding". It is intended for development and for embedding into other systems,
and uses Google's [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2).

- Ingestion (split into time windows with ffmpeg, embed, store in a vector DB) and search run in **one container**
- Video, audio, images and text are searched in **one shared vector space** (text to video, image to video, audio to audio, and so on)
- Includes a **WebUI** for development (`http://localhost:8000/`). API reference at **`/scalar`** (Scalar, browse and try calls), `/docs` (Swagger UI) and [docs/openapi.json](docs/openapi.json)
- Embeds **RTSP streams** (network cameras, etc.) while receiving them, and **windows become searchable immediately** (notifications for new matching windows are also available)
- Built-in **MCP server** (`http://localhost:8000/mcp`). AI agents can search and ingest ([docs/mcp.md](docs/mcp.md))
- Supports CPU / NVIDIA GPU / AMD GPU / Intel GPU (just pick the image)
- Images are built by GitHub Actions. Users only need **`docker pull` / `docker run`**

The other documents in `docs/` are written in Japanese.

> ## Before you use it (at your own risk)
> - **Authentication is disabled by default**, so that it starts without extra configuration. If you use it outside a trusted network, always set `API_TOKEN` and put it behind a reverse proxy that terminates TLS (HTTPS). The app itself does not encrypt traffic.
> - Recorded video and audio may contain **personal information** (faces, voices, license plates, etc.). What is stored is "vectors, thumbnails and references to the original files", but if `API_TOKEN` is not set, anyone can access the APIs that serve thumbnails and the original video. Handling this data is the user's responsibility.
> - The security measures in place and the remaining risks are described in [docs/security.md](docs/security.md).
> - This is for development and evaluation. Use in production is at your own risk.

## Quick start (CPU first)

```bash
docker run -d --name mediasearch -p 8000:8000 -v ./data:/data \
  ghcr.io/miyabiver39/embeddinggemma-poc:cpu
```

Open `http://localhost:8000/`. Choose a video on the "取り込み" (Ingest) tab, then type words in Japanese or English on the "検索" (Search) tab.
(The model is bundled in the image, so nothing is downloaded at startup. Loading the model takes a few tens of seconds.)

> If the pull is denied: the repository administrator must set the visibility of each image to **Public** in the GitHub Packages settings.

### Ingest a recording directory as is

```bash
docker run -d --name mediasearch -p 8000:8000 -v ./data:/data -v /path/to/recordings:/recordings:ro \
  ghcr.io/miyabiver39/embeddinggemma-poc:cpu

# Ingest one file (the recording start time is read from 20260101_090000 in the file name)
curl -X POST localhost:8000/api/ingest/path -H 'Content-Type: application/json' \
  -d '{"path":"/recordings/group-a/20260101_090000.mp4","group_id":"group-a","location":"entrance"}'

# Ingest a whole folder (folder names such as group-a become the group ID; files already ingested are skipped)
curl -X POST localhost:8000/api/ingest/dir -H 'Content-Type: application/json' \
  -d '{"dir":"/recordings","group_from_dir":true}'
```

- If `start_ts` is omitted, the date and time in the file name (`20260101_090000`, `2026-01-01T09-00-00`, etc.) is used as the recording start time. If it cannot be read, the time of the request is used.
- Specifying the same file again does not register it twice (the existing source is returned). To ingest it again, add `"force": true`.
- Files are checked with ffprobe when the request is accepted, and files that cannot be handled as video or audio are rejected with 422 (see `docs/operations.md` for the accepted extensions).

### Ingest an RTSP stream while receiving it

```bash
curl -H 'Content-Type: application/json' \
  -d '{"url": "rtsp://user:pass@192.168.1.10:554/stream1", "group_id": "entrance"}' http://localhost:8000/api/streams
curl http://localhost:8000/api/streams        # Receiving status (windows created, delay, errors). Passwords are shown as ***
```

Windows can be searched as soon as they are created. You can also be notified of new windows that match a text (`GET /api/search/live`, [docs/api.md](docs/api.md)).
Registrations survive a server restart, and the stream reconnects automatically when disconnected.

### Watch a recording folder and ingest automatically

```bash
docker run -d --name mediasearch -p 8000:8000 -v ./data:/data -v /path/to/recordings:/recordings:ro \
  -e WATCH_DIRS=/recordings ghcr.io/miyabiver39/embeddinggemma-poc:cpu
```

The folders in `WATCH_DIRS` are scanned periodically, and new recording files are ingested automatically.
To avoid files that are still being written, only files whose last modification is at least `WATCH_SETTLE_SEC` seconds old (default 30) are picked up.
The name of the folder containing the file becomes the group ID (`group-a` for `/recordings/group-a/...`). The status is shown on the "状態" (Status) tab of the WebUI.

## Image variants

| Tag | Use | Run options |
|---|---|---|
| `:cpu` (= `:latest`) | CPU inference. Runs anywhere | none |
| `:cuda` | NVIDIA GPU (RTX 3060, etc.) | `--gpus all` |
| `:rocm` | AMD GPU (RX 9060 XT, etc.) | `--device=/dev/kfd --device=/dev/dri --group-add video` |
| `:intel` | Intel GPU (Arc / Core Ultra integrated) | `--device=/dev/dri` |
| `:slim` | No model. App only, embedding is delegated to an external compute container | none |

```bash
# NVIDIA
docker run -d -p 8000:8000 -v ./data:/data --gpus all ghcr.io/miyabiver39/embeddinggemma-poc:cuda
# AMD
docker run -d -p 8000:8000 -v ./data:/data --device=/dev/kfd --device=/dev/dri --group-add video \
  ghcr.io/miyabiver39/embeddinggemma-poc:rocm
# AMD (Docker Desktop on Windows / WSL2: use /dev/dxg instead of /dev/kfd)
docker run -d -p 8000:8000 -v ./data:/data --device=/dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro \
  ghcr.io/miyabiver39/embeddinggemma-poc:rocm
# Intel
docker run -d -p 8000:8000 -v ./data:/data --device=/dev/dri ghcr.io/miyabiver39/embeddinggemma-poc:intel
```

You can check whether the GPU is used on the "状態" (Status) tab of the WebUI, or with `embedder.accelerator` in `curl localhost:8000/api/info`
(`nvidia-cuda` / `amd-rocm` / `intel-xpu` / `cpu`).
**Of the GPU images, only NVIDIA RTX 3060 (Linux) and AMD RX 9060 XT (Windows WSL2) have been tested on real hardware** (Intel and native Linux AMD are not tested yet). See [docs/gpu.md](docs/gpu.md) for the test procedure, results and known issues.
Older Intel integrated GPUs (Iris Xe, etc.) are likely not supported by PyTorch XPU; in that case the image runs on the CPU.

## Benchmark

Measured with the bundled benchmark tool (`bench/`) on synthetic video, audio and images (the defaults of `bench/make_samples.py`).
Ingestion figures are the median of 3 runs. See [bench/README.md](bench/README.md) (Japanese) for the detailed conditions and how to read the results.

| Measurement | CPU (Ryzen 7 5700X, `:cpu`) | RTX 3060 12GB (`:cuda`) | RX 9060 XT 16GB (`:rocm`) |
|---|---|---|---|
| Video ingestion (1080p, 60 s, 30 two-second windows) | 242.7 s (8.09 s per window) | **5.2 s (0.17 s per window)** | 7.8 s (0.26 s per window) |
| Ingestion speed relative to real time | 0.25x | **11.5x** | 7.7x |
| Audio ingestion (60 s, 6 ten-second chunks) | 5.71 s | **0.70 s** | 0.91 s |
| Image ingestion (1080p, 50 images) | 0.28 images/s | **9.67 images/s** | 5.62 images/s |
| Text search (4 concurrent, 30 s) | 8.1 req/s (p50 497 ms, p95 524 ms) | **194 req/s (p50 19 ms)** | 16.9 req/s (p50 176 ms) |

- CPU (`:cpu`): Windows 11 Pro + Docker Desktop (WSL2, 16 logical CPUs), AMD Ryzen 7 5700X. Re-measured on 2026-11-21 with the CI-built `:cpu` image (`main` at `47f1668`, PyTorch 2.14.1+cpu, transformers 5.23.0, sentence-transformers 6.3.0). Startup (including warmup) took 11.8 s; single-request search responded in p50 134 ms (7.53 req/s); search during video ingestion kept p50 504 ms / p95 548 ms (p99 8.0 s, waiting at most one in-flight window). Compared with earlier CPU measurements (`056b937`: 4.57 req/s, p95 1,890 ms; 2026-10-13: 7.1 req/s, p95 1,297 ms), concurrent text search throughput rose and p95 latency fell sharply. With `SKIP_STATIC` on a synthetic 8-scene x 8 s video (64 s, 32 windows), 25 windows were skipped and ingestion dropped from 257.8 s (0.25x) to 56.8 s (1.13x real time); ingesting 100 document chunks (~76,000 chars) took 72.5 s (725 ms per chunk) ([bench/README.md](bench/README.md)).
- RTX 3060: Ubuntu 26.04.1 LTS, Core i7-6700K, driver 595.99.02. Measured on 2026-11-20 with the CI-built `:cuda` image (main at c5a27c3), which includes the inference optimizations (torch.compile, CUDA Graphs and more; see "推論の高速化" in [docs/gpu.md](docs/gpu.md)) and the improvements from PR #14. Speed before and after PR #14 was the same within measurement error ([bench/README.md](bench/README.md)). Before the optimizations (2026-11-19): video 9.4 s (0.31 s per window), audio 1.01 s, images 5.12 images/s, search 17.4 req/s (p50 166 ms). A single search responds in p50 7.6 ms (51 ms before). Peak GPU memory use is about 3.5 GB, and video is also decoded on the GPU. The first start takes about 80 seconds to prepare the optimizations (about 37 seconds afterwards).
- The RTX 3060 audio figure is after removing the audio-side optimization (torch.compile) for quality reasons (0.44 s before; see [docs/gpu.md](docs/gpu.md)).
- RX 9060 XT: Windows 11 + Docker Desktop (WSL2). Measured on 2026-10-13 with the fast attention kernels enabled (the default of the `rocm` image; [docs/gpu.md](docs/gpu.md)).
- On a GPU, two-second-window video is ingested more than 6 times faster than real time. Most of the time is spent on inference; audio and short text searches need little computation, so the GPU makes less of a difference there.
- These figures are for synthetic media. With real recordings they vary with resolution, length and window settings.

### Windows benchmark with build `056b937`

Measured on 2026-11-19 and 2026-11-20 (JST) on the same Windows 11 Pro PC with Docker Desktop 4.96.3 (WSL2, 16 logical CPUs).
Both public images used `ROLE=all` and `google/embeddinggemma-2` (768 dimensions). Build `056b937` predates the change that enables fast ROCm attention kernels by default.

| Inference hardware | Image | Precision | Maximum batch size |
|---|---|---|---:|
| AMD Ryzen 7 5700X (8 cores, 16 threads) | `:cpu` | FP32 | 1 |
| AMD Radeon RX 9060 XT (16 GB, via `/dev/dxg`) | `:rocm` | BF16 | 4 |

| Measurement | CPU | GPU | GPU speed relative to CPU |
|---|---:|---:|---:|
| Video ingestion (1080p H.264, 30 fps, 60 s, 30 windows) | 245.0 s (8.17 s per window) | 30.19 s (1.01 s per window) | 8.1× |
| Audio ingestion (60 s, 6 ten-second chunks) | 5.96 s | 0.99 s | 6.0× |
| Image ingestion (50 1080p JPEGs) | 0.225 images/s | 1.921 images/s | 8.5× |
| Text search (4 concurrent requests) | 4.57 req/s, p50 724 ms, p95 1,890 ms | 14.33 req/s, p50 159 ms, p95 771 ms | 3.1× (req/s) |

- Both configurations decoded video on the CPU and embedded two frames per two-second window, with zero overlap and video-only input. Ingestion reached 0.24× real time on the CPU and 1.99× real time on the GPU.
- After warm-up, video and audio ingestion were each measured three times and reported as medians. Image ingestion was measured once per configuration. Text search used the default queries with `top_k=10` and an index of 86 entries. Each search metric is the median of three 30-second trials. Search throughput ranged from 4.30 to 6.30 req/s on the CPU and from 7.50 to 14.37 req/s on the GPU.

The RX 9060 XT 16 GB optimization raised repeated text search from 26.1 to 613.2 requests/s at four concurrent requests. Total video ingestion and first-time text search were roughly unchanged. The tested vectors matched bit for bit. [RX 9060 XT benchmark](bench/README.md#rx-9060-xt-16-gb-のローカル推論の高速化)

## Three ways to run (ROLE)

| ROLE | Contents | When to use |
|---|---|---|
| `all` (default) | Ingestion + search + DB + WebUI + embedding | Everything in one container |
| `app` | Ingestion + search + DB + WebUI (no model) | Delegate embedding to a GPU on another machine |
| `compute` | Embedding API only (`/compute/*`) | Run on the GPU server |

```bash
# GPU server
docker run -d -p 8001:8000 --gpus all -e ROLE=compute ghcr.io/miyabiver39/embeddinggemma-poc:cuda
# Lightweight machine
docker run -d -p 8000:8000 -v ./data:/data -e EMBEDDING_URL=http://gpu-server:8001 \
  ghcr.io/miyabiver39/embeddinggemma-poc:slim
```

`docker compose --profile split up` starts the same setup ([docker-compose.yml](docker-compose.yml)).
For access from outside, a setup with Nginx terminating HTTPS is `docker compose -f deploy/compose.nginx.yml up -d` ([docs/nginx.md](docs/nginx.md)).
Video processing (frame extraction, converting audio to 16 kHz) always happens on the app side, so **only images and PCM audio** are sent to compute.
You can also send your own pre-processed frames to `/api/ingest/frames`.

## Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `ROLE` | `all` (`app` for slim) | `all` / `app` / `compute` |
| `EMBEDDING_BACKEND` | `local` (`remote` for app) | `local` in-process / `remote` external compute / `dummy` for testing |
| `EMBEDDING_URL` | `http://localhost:8001` | URL of compute when remote |
| `DEVICE` | `auto` | `auto` / `cpu` / `cuda` (AMD ROCm is also `cuda`) / `xpu` |
| `DTYPE` | `auto` | bf16 on GPU, fp32 on CPU. Set `float32` if the output becomes NaN |
| `DIMS` | `768` | Dimensions of stored vectors (768/512/256/128). **Cannot be changed after the DB is created** (see below) |
| `WINDOW_SEC` | `2` | Length of the time window turned into one vector (seconds) |
| `FRAMES_PER_WINDOW` | `2` | Frames per window |
| `OVERLAP_SEC` | `0` | Overlap between windows (seconds) |
| `INCLUDE_AUDIO` | `false` | Include the video's audio in the same vector |
| `SKIP_STATIC` | `false` | Do not embed windows whose video has not changed from the previous window; extend the previous window instead (video-only windows; both file ingestion and RTSP streams; can be changed per ingestion or stream registration with `skip_static`). Reduces the number of windows and the inference load for cameras with little motion ([docs/design.md](docs/design.md)) |
| `STATIC_THRESHOLD` | `4.0` | For `SKIP_STATIC`, the largest per-block brightness difference (0-255) still treated as "no change". Smaller values skip less |
| `DOC_CHUNK_CHARS` | `800` | Approximate number of characters per document vector (100-4000). Chunks never span pages |
| `DOC_OVERLAP_CHARS` | `100` | Number of characters from the end of a chunk repeated at the start of the next one (0 to half of `DOC_CHUNK_CHARS`) |
| `AUDIO_CHUNK_SEC` | `10` | Chunk length for audio files (seconds) |
| `IMAGE_MAX_SIDE` | `448` | Long side images are resized to (px) |
| `FFMPEG_HWACCEL` | `auto` | GPU used to decode video (`auto` / `cuda` / `vaapi` / `qsv` / `none`). Falls back to the CPU if unavailable (`docs/gpu.md`) |
| `FFMPEG_HWACCEL_DEVICE` | (empty) | Device for GPU decoding (e.g. `/dev/dri/renderD128`). Automatic if empty |
| `FFMPEG_SKIP_FRAMES` | `noref` | Frames skipped when decoding. `noref` (skip non-reference frames; timing is off by 1-2 frames) / `none` (all frames) / `keyframes` (keyframes only; several times faster, but timing can be off by a few seconds) |
| `EMBED_BATCH` | `auto` | Maximum number of items embedded together. `auto` is 1 on CPU (no batching; measured not to be faster on CPU) and 4 on GPU (small effect, see `docs/gpu.md`) |
| `TORCH_COMPILE` | `auto` | Inference optimizations (torch.compile, CUDA Graphs). `auto` uses them on NVIDIA GPUs when a C compiler is available (the `cuda` image). `off` disables them (faster startup, slower inference). The prepared results are cached in `DATA_DIR/cache/torch` (`docs/gpu.md`) |
| `TORCH_CACHE_DIR` | `DATA_DIR/cache/torch` | Persistent directory for PyTorch (`TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR`) compilation caches (explicit per-variable overrides take precedence) |
| `WARMUP_ON_START` | `true` | Runs one dummy warmup inference (text, image, audio) right before `/readyz` becomes ready (skipped for `remote`, or when already warmed up by `TORCH_COMPILE`) |
| `INGEST_PREFETCH` | `2` | Bounded queue capacity (1-16) for the video ingestion producer-consumer pipeline. Overlaps frame decoding and preprocessing for window N+1 with inference of window N while applying backpressure to bound memory usage |
| `INGEST_WORKERS` | `1` | Number of files ingested at the same time (1-8). Decoding, preprocessing and saving run in parallel; inference still goes to the GPU one call at a time through the priority gate, so search priority and vectors are unchanged. Useful on fast GPUs where a single ffmpeg decoder cannot keep the GPU busy. Memory use grows with this number (prepared inputs waiting for the GPU are capped server-wide at `INGEST_PREFETCH` + 2 batches regardless of this number, and the GPU images default to `MALLOC_ARENA_MAX=2` when it is 2 or more; about 4.6 GB peak with 8 files at once on an RTX 3060 with 6-second windows) |
| `PIN_MEMORY` | `true` | Put inputs for the GPU in pinned (page-locked) memory. `false` uses less memory (about 4.6 GB → 3.9 GB peak with 8 files at once on an RTX 3060) at a cost of 0-2% ingestion speed on that GPU |
| `IMAGE_MAX_TOKENS` | `0` | Token limit per image (0 = model default) |
| `TOP_K` | `10` | Default number of search results |
| `DATA_DIR` | `/data` | Where the DB, thumbnails and uploaded videos are stored |
| `STORE_MEDIA` | `true` | If `false`, uploaded videos and images are deleted after ingestion and no thumbnails are created. Search results then contain only time, score and similar information (can also be set per ingestion with `store_media`) |
| `WATCH_DIRS` | (empty) | Folders to watch and ingest automatically (comma-separated). Disabled if empty |
| `WATCH_INTERVAL_SEC` | `60` | Interval for scanning the watched folders (seconds) |
| `WATCH_SETTLE_SEC` | `30` | Only ingest files whose last modification is at least this many seconds old (avoids files being written) |
| `WATCH_GROUP_FROM_DIR` | `true` | Use the name of the folder containing the file as the group ID |
| `WATCH_PRESET` | (empty) | Preset for ingesting watched folders (`object` / `action` / `speech`). If empty, the window defaults above are used |
| `API_TOKEN` | (empty) | If set, `/api` and `/compute` require a token (`docs/security.md`) |
| `API_READ_TOKEN` | (empty) | Token that only allows search and reading (no ingestion, changes or deletion). Only effective when `API_TOKEN` is set |
| `MAX_PENDING_SEARCHES` | `32` | Maximum number of searches processed at once. Excess requests are refused with 503 (`Retry-After: 5`) to keep the API responsive. `0` for no limit |
| `EMBEDDING_TOKEN` | same as `API_TOKEN` | Token sent to compute when remote |
| `INGEST_ROOTS` | `/recordings` | Folders allowed for path and folder ingestion (comma-separated). `DATA_DIR` and the watched folders are added automatically |
| `MAX_UPLOAD_MB` | `4096` | Maximum size of one request (MB). `0` for unlimited |
| `CLIP_MAX_SEC` | `600` | Maximum length (seconds) of one clip cut out with `GET /api/sources/{id}/clip`. `0` disables clipping |
| `ALLOWED_ORIGINS` | (empty) | Origins allowed to call state-changing APIs from a page on another origin (comma-separated) |
| `ALLOWED_HOSTS` | `*` | Accepted Host headers (comma-separated). Protection against DNS rebinding |
| `PUID` / `PGID` | `1000` | User / group ID the app runs as. `PUID=0` keeps it running as root |
| `RETENTION_DAYS` | `0` | Delete windows older than this many days (based on recording time), together with their vectors and thumbnails. `0` keeps everything. Can also be set per stream with `retention_hours` |
| `MIN_FREE_DISK_MB` | `1024` | Stop new ingestion when free space in `DATA_DIR` falls below this (507; search and deletion keep working). `0` disables the check |
| `JOB_RETRIES` | `3` | Number of retries for temporary failures such as compute being unreachable or low disk space (starting at 30 seconds, doubling each time) |
| `JOB_HISTORY_DAYS` | `30` | Days to keep the history of finished jobs (the latest job of each source is kept). `0` keeps everything |
| `BACKUP_DIR` | `DATA_DIR/backups` | Where backups are stored. Mounting another disk or a NAS is recommended in case of disk failure (`docs/operations.md`) |
| `BACKUP_INTERVAL_HOURS` | `0` | Take an online backup automatically at this interval (e.g. `24`). `0` disables automatic backups (use `POST /api/admin/backup`) |
| `BACKUP_KEEP` | `7` | Number of backups to keep (oldest deleted first). `0` keeps all |
| `MAX_STREAMS` | `8` | Number of RTSP streams that can be received at the same time |
| `TZ` | `Asia/Tokyo` | Time zone for interpreting times (dates in file names, `start_ts` without a time zone) |
| `PUBLIC_URL` | (empty) | URL of this server as seen from outside (e.g. `https://search.example.com`). Makes thumbnail and video URLs absolute in notification webhook bodies |
| `LOG_LEVEL` | `INFO` | Minimum log level (`DEBUG` / `INFO` / `WARNING` / `ERROR`) |
| `LOG_FORMAT` | `text` | `json` outputs one JSON object per line (for log collectors; also outputs access logs with request ID and duration) |

Ingestion presets (`preset`): `object` (2-second windows, 2 frames, no audio; for objects and scenes) / `action` (4-second windows, 4 frames, no audio; for actions) / `speech` (6-second windows, 6 frames, with audio; when speech or sound matters).

## Design rules you should know

- **Metadata (group ID, location, time) is not embedded into vectors.** It is stored in separate columns and used to filter at search time (mixing it into vectors lowers search accuracy and reproducibility).
- The DB records the **model ID, dimensions and window settings**, and **adding data with different settings is rejected with 409** (to prevent mixing vector spaces). To change `DIMS`, recreate `DATA_DIR` and ingest again.
- "Video only (frames)", "video + audio (tav)", "audio only (audio)" and "still image (image)" are stored as separate kinds (`kind`) and selected at search time. With `auto`, text and image searches look at tav or frames (depending on `INCLUDE_AUDIO`) and image; audio searches look at `audio`.
- **There is a mode that does not keep videos or images.** When ingesting with `STORE_MEDIA=false` (or `store_media=false` per ingestion), only vectors and information (source name, time, group ID, etc.) are kept; uploaded files are deleted after ingestion and no thumbnails are created. `media_url` / `thumb_url` in search results become `null`. Files ingested by path belong to the user, so they are not deleted, only excluded from serving. Since the original file is gone, re-ingestion (reindex) is not possible.
- **Images alone can be registered.** When frames cannot be extracted from video, or when only photos are handled, each image is ingested as one item (`/api/ingest/image`, path, folder and watched folder ingestion are all supported). The capture time is taken from the specified value, the date in the file name, then the EXIF capture time, in that order.
- **Documents (PDF, Word, text) can be registered.** `.pdf` / `.docx` / `.txt` / `.md` files are split into chunks of about 800 characters, one vector per chunk (`/api/ingest/document`, path, folder and watched folder ingestion are all supported). Search them with `kind=document` (or `all` to mix them with video); results include the page (`page`) and the chunk text (`text`). Shift_JIS text files are supported. Image-only PDFs (scans) cannot be ingested because no text can be extracted. Documents are not subject to the retention period (`RETENTION_DAYS`).
- **A retention period can be set.** With streams and watched folders, windows keep increasing (one 2-second-window stream produces about 43,000 windows and about 140 MB of vectors per day). With `RETENTION_DAYS`, old windows are deleted every 10 minutes, keeping memory and disk usage within bounds. Files whose windows are all gone get the status `expired`; the original files are not deleted.
- **Vectors are stored in an embedded vector DB (USearch).** Settings, sources and window information are stored in SQLite (`mediasearch.db`), and vectors in `vectors.usearch`. Both are single files in `DATA_DIR`; no separate server is needed. As the count grows it switches to approximate search (HNSW), so search time stays nearly constant. Memory is proportional to the vector size (about 320 MB for 100,000 windows at 768 dimensions).

## API overview

| | |
|---|---|
| `GET /api/info` | Status, version, defaults, presets, index counts, watched folder status |
| `POST /api/ingest/video` / `audio` / `image` / `document` | Ingest uploaded files (several images can be sent at once; documents are PDF, Word or text) |
| `POST /api/ingest/path` | Ingest a path inside the container |
| `POST /api/ingest/dir` | Ingest a folder inside the container in bulk |
| `POST /api/watch/scan` | Scan the watched folders now |
| `POST /api/ingest/frames` | Ingest pre-processed frames (+ audio) |
| `POST /api/search/text` / `image` / `audio` | Search (filter by group ID, location, period, kind, minimum score) |
| `GET /api/jobs`, `/api/sources`, `/api/sources/{id}` and more | View, delete and re-ingest jobs and sources |
| `POST /api/streams`, `GET /api/streams` and more | Register RTSP streams, check status, stop, resume, delete |
| `GET /api/search/live` | Notify windows added during ingestion that match a text (real-time search, Server-Sent Events) |
| `GET /healthz` / `GET /readyz` | Liveness (is the process running) / readiness (DB, vector DB, compute, free space; 503 if it cannot serve). No authentication required |
| `PATCH /api/sources/{id}` | Change a source's name, group ID and location (no re-ingestion needed) |
| `POST /api/jobs/{id}/cancel` | Cancel an ingestion job (a running job stops at a window boundary) |
| `GET /api/metrics` | Resource usage (CPU, memory, disk, GPU) and inference utilization. Read by the benchmark (`bench/`) |
| `GET /api/stats` | Summary of ingestion processing time (per-job breakdown in `timings` of `GET /api/jobs/{id}`; measured with `bench/bench.py`) |
| `GET /api/media/{id}`, `/api/thumb/{id}` | Original video (Range supported) and thumbnails |
| `/compute/*` | Embedding API (ROLE=compute / all) |
| `/mcp` | MCP server (for AI agents; ROLE=all / app) |

For details see `/scalar` (Scalar), `/docs` (Swagger UI), `/redoc`, [docs/api.md](docs/api.md) (developer guide) and [docs/openapi.json](docs/openapi.json) (OpenAPI specification).

## Documentation (in Japanese)

- [docs/design.md](docs/design.md) — Design (architecture, window design, DB, API, plan for native runtimes)
- [docs/api.md](docs/api.md) — API guide (authentication, errors, examples, client generation)
- [docs/mcp.md](docs/mcp.md) — MCP server (use from AI agents)
- [docs/operations.md](docs/operations.md) — Operations notes (capacity estimates, tuning, troubleshooting)
- [docs/nginx.md](docs/nginx.md) — Running behind Nginx (HTTPS, per-client limits, access logs; `deploy/`)
- [docs/gpu.md](docs/gpu.md) — GPU mode (RTX 3060 / RX 9060 XT / Intel) test procedure
- [docs/release.md](docs/release.md) — Release procedure and the attached SBOM and vulnerability scan results
- [docs/security.md](docs/security.md) — Security measures and remaining risks
- [bench/README.md](bench/README.md) — Benchmark tool (speed, resource usage, bottleneck diagnosis)
- [SECURITY.md](SECURITY.md) — How to report vulnerabilities
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) — Third-party software and licenses
- [AGENTS.md](AGENTS.md) — Guide for AI agents

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # For tests without the model
pytest -q                        # End-to-end tests with dummy embeddings (requires ffmpeg)
pip install -e ".[model]"        # To use the real model (transformers 5.23.0 or later; the images pin 5.23.0)
RUN_MODEL_TESTS=1 pytest -q -m model
EMBEDDING_BACKEND=dummy DATA_DIR=./data uvicorn --factory mediasearch.main:create_app --reload
```

## Releases, SBOM and vulnerability scans

Pushing a tag such as `v0.2.0` publishes versioned images (`:0.2.0-cpu`, etc.) and creates a GitHub release
with the **SBOM (SPDX / CycloneDX) and vulnerability scan results** of all images attached.
The latest main images are scanned every week, and the results are kept in the GitHub Actions run. See [docs/release.md](docs/release.md) for details.

## Security

- If you find a vulnerability, please report it through GitHub's private reporting feature instead of a public issue (see [SECURITY.md](SECURITY.md)).
- See [docs/security.md](docs/security.md) for the measures in place and the settings required on the user's side.
- The SBOM and vulnerability scan results of the published images are attached to each release. The "vulnerabilities" badge above shows the result of the weekly scan of the latest main images (the highest count among all images).

## License

- Source code in this repository: [Apache License 2.0](LICENSE)
- Model (EmbeddingGemma 2, Google): Apache-2.0 (according to the model card). Bundled in the published images.
- The published images contain open source software such as FFmpeg (GPL-2.0 or later) and PyTorch, and the GPU images also contain
  libraries under proprietary licenses from NVIDIA (CUDA) and Intel (oneAPI). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the list.
