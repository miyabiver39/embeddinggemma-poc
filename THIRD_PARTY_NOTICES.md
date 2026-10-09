# サードパーティのソフトウェアとライセンス

本リポジトリのソースコードは [Apache License 2.0](LICENSE) で提供します。
公開しているコンテナイメージには、以下のサードパーティのソフトウェアとモデルが含まれます。それぞれのライセンスに従ってご利用ください。

- 下表は主要なものの抜粋です。**全件**は、各リリースに添付している SBOM(`sbom-<イメージ>.spdx.json` / `sbom-<イメージ>.cdx.json`)に、パッケージごとのライセンスとともに記載しています。
- ライセンスの表記は、各パッケージのメタデータと SBOM の記載によります(2026-10-08 時点のイメージで確認)。正式な条件は、それぞれの配布元のライセンス文をご確認ください。

## すべてのイメージに含まれるもの

| ソフトウェア | 用途 | ライセンス |
|---|---|---|
| Ubuntu 24.04(ベースイメージの OS パッケージ) | 実行環境 | パッケージごとに異なる(主に GPL / LGPL / MIT / BSD。各パッケージの `/usr/share/doc/<名前>/copyright`) |
| FFmpeg(Ubuntu のパッケージ) | 映像・音声の分割と変換 | **GPL-2.0 以降**(Ubuntu のパッケージは GPL の部品を有効にしてビルドされているため、全体として GPL) |
| Python 3.12(Ubuntu のパッケージ) | 実行環境 | PSF-2.0 |
| FastAPI | Web API | MIT |
| Starlette | Web API | BSD-3-Clause |
| Uvicorn | HTTP サーバ | BSD-3-Clause |
| Pydantic | 入力の検証 | MIT |
| python-multipart | アップロードの解析 | Apache-2.0 |
| NumPy | 数値計算 | BSD-3-Clause ほか(同梱ライブラリを含む) |
| Pillow | 画像処理 | MIT-CMU |
| HTTPX | compute との通信 | BSD-3-Clause |
| USearch(`usearch`)と NumKong(`numkong`) | ベクトル DB(近傍検索) | Apache-2.0 |
| MCP Python SDK(`mcp`)と依存パッケージ(httpx2、sse-starlette など) | MCP サーバー | MIT(依存パッケージは BSD-3-Clause / MIT など) |

## モデルを同梱するイメージ(cpu / cuda / rocm / intel)

| ソフトウェア | 用途 | ライセンス |
|---|---|---|
| [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)(Google) | 埋め込みモデル(イメージに同梱。版は `docker/Dockerfile` の `MODEL_REVISION`) | Apache-2.0(モデルカードの記載による) |
| PyTorch | 推論 | Apache-2.0 ほか(パッケージのメタデータによる。同梱ライブラリのライセンスはパッケージ内の記載を参照) |
| torchvision | 画像の前処理 | BSD-3-Clause |
| Transformers(Hugging Face。固定したコミットのアーカイブ) | モデルの実装 | Apache-2.0 |
| sentence-transformers | 埋め込みの計算 | Apache-2.0 |
| huggingface_hub / tokenizers / safetensors | モデルの読み込み | Apache-2.0 |
| SoundFile | 音声の読み込み | BSD-3-Clause |

## GPU 用のイメージに追加で含まれるもの

| イメージ | ソフトウェア | ライセンス |
|---|---|---|
| `cuda` | NVIDIA CUDA ランタイム・cuBLAS・cuDNN・NCCL など(`nvidia-*` パッケージ。PyTorch が利用) | **NVIDIA の独自ライセンス**(NVIDIA Software License Agreement / CUDA EULA) |
| `cuda` | Triton | MIT |
| `rocm` | AMD ROCm のライブラリ(PyTorch の ROCm 版に同梱) | 各ライブラリのライセンス(PyTorch のパッケージ内の記載を参照) |
| `rocm` | Triton(ROCm 版) | MIT |
| `intel` | Intel oneAPI の実行環境(DPC++ / SYCL ランタイム、OpenMP、OpenCL ランタイムなど) | **Intel End User License Agreement for Developer Tools** |
| `intel` | Intel oneMKL、Intel MPI の実行環境 | **Intel Simplified Software License** |
| `intel` | Intel GPU ドライバ(Ubuntu の `libze-intel-gpu1`、`intel-opencl-icd`)、Level Zero | MIT / BSD-3-Clause ほか |
| `intel` | Triton(XPU 版) | MIT |

GPU 用のイメージに含まれる NVIDIA と Intel のライブラリは、オープンソースではない独自のライセンスで提供されています。
再配布や商用での利用には、それぞれの条件が適用されます。

## FFmpeg(GPL)のソースコード

イメージに含まれる FFmpeg は、Ubuntu 24.04(noble)の公式パッケージです。ソースコードは Ubuntu から入手できます。

- Launchpad: https://launchpad.net/ubuntu/+source/ffmpeg
- コンテナ内での確認: `dpkg -s ffmpeg`(版)、`/usr/share/doc/ffmpeg/copyright`(ライセンス文)

## ブラウザが読み込むもの(イメージには含まれません)

| ソフトウェア | 用途 | ライセンス |
|---|---|---|
| Scalar(`@scalar/api-reference`。jsDelivr から版を固定して読み込み) | API リファレンスの画面(`/scalar`) | MIT |
| Swagger UI / ReDoc(FastAPI の既定。jsDelivr から読み込み) | API 仕様の画面(`/docs`、`/redoc`) | Apache-2.0 / MIT |

## ビルドと検査に使うツール(イメージには含まれません)

| ツール | 用途 | ライセンス |
|---|---|---|
| syft / grype(Anchore) | SBOM の作成と脆弱性の検査 | Apache-2.0 |
| CodeQL(GitHub) | ソースコードの静的解析 | GitHub CodeQL Terms and Conditions(公開リポジトリでは無償) |
| OpenSSF Scorecard | リポジトリの運用の評価 | Apache-2.0 |
| pytest / pytest-cov / Ruff | テストと静的チェック | MIT |
| MediaMTX(`bluenviron/mediamtx`) | RTSP の受信の確認用サーバー(`docs/operations.md` の手順のみ) | MIT |
