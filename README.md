# mediasearch — EmbeddingGemma 2 で動画・音声・画像・文章を横断検索する(開発用)

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

「赤い車が映っている場面」「サイレンが鳴っている区間」のような**言葉(または画像・音声)**で、録画映像の**該当時刻**を探すための
バックエンド基盤です。他のシステムへの組み込みを想定した開発用で、
Google の [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) を使います。

- 取り込み(ffmpeg で時間窓に分割 → ベクトル化 → SQLite に保存)と検索を **1つのコンテナ**で提供
- 動画・音声・画像・テキストを**同じベクトル空間**で検索(テキスト→映像、画像→映像、音声→音声 など)
- 開発確認用の **WebUI** 付き(`http://localhost:8000/`)、API 仕様は `/docs`(Swagger UI)
- CPU / NVIDIA GPU / AMD GPU / Intel GPU に対応(イメージを選ぶだけ)
- ビルドは GitHub Actions が行います。利用者は **`docker pull` / `docker run` だけ**です

> ## ご利用前の注意事項(自己責任)
> - **既定では認証が無効です。** 追加の設定なしで起動できるようにするためです。信頼できるネットワークの外で使う場合は、必ず `API_TOKEN` を設定し、TLS(HTTPS)を終端するリバースプロキシの内側に置いてください。アプリ自体は暗号化を行いません。
> - 録画映像・音声には**個人情報**(顔・声・車のナンバーなど)が含まれ得ます。保存されるのは「ベクトル・サムネイル・元ファイルの参照」ですが、`API_TOKEN` を設定していない場合、サムネイルや元動画の配信 API には誰でもアクセスできます。取り扱いは利用者の責任です。
> - 実施しているセキュリティ対策と残っているリスクは [docs/security.md](docs/security.md) にまとめています。
> - 開発・検証用途です。運用での利用は自己責任でお願いします。

## クイックスタート(まずは CPU)

```bash
docker run -d --name mediasearch -p 8000:8000 -v ./data:/data \
  ghcr.io/miyabiver39/embeddinggemma-poc:cpu
```

`http://localhost:8000/` を開きます。「取り込み」タブで動画を選び、「検索」タブで日本語・英語の言葉を入力します。
(モデルを同梱しているため、起動時のダウンロードはありません。起動時のモデル読み込みに数十秒かかります。)

> pull が拒否される場合: リポジトリ管理者が GitHub の Packages 設定で、各イメージの公開範囲を **Public** にする必要があります。

### 録画ディレクトリをそのまま取り込む

```bash
docker run -d --name mediasearch -p 8000:8000 -v ./data:/data -v /path/to/recordings:/recordings:ro \
  ghcr.io/miyabiver39/embeddinggemma-poc:cpu

# 1 ファイルを取り込む(録画開始時刻はファイル名の 20260101_090000 から読み取ります)
curl -X POST localhost:8000/api/ingest/path -H 'Content-Type: application/json' \
  -d '{"path":"/recordings/cam01/20260101_090000.mp4","camera_id":"cam01","location":"玄関"}'

# フォルダをまとめて取り込む(フォルダ名 cam01 などをカメラ ID にする。取り込み済みのファイルは対象外)
curl -X POST localhost:8000/api/ingest/dir -H 'Content-Type: application/json' \
  -d '{"dir":"/recordings","camera_from_dir":true}'
```

- `start_ts` を省略すると、ファイル名に含まれる日時(`20260101_090000`、`2026-01-01T09-00-00` など)を録画開始時刻にします。読み取れない場合は受付時刻になります。
- 同じファイルを再度指定しても重複して登録しません(既存の取り込み元を返します)。取り込み直す場合は `"force": true` を付けます。
- 受付の時点で ffprobe による検証を行い、映像・音声として扱えないファイルは 422 で拒否します(取り込める拡張子は `docs/operations.md` を参照)。

### 録画フォルダを監視して自動で取り込む

```bash
docker run -d --name mediasearch -p 8000:8000 -v ./data:/data -v /path/to/recordings:/recordings:ro \
  -e WATCH_DIRS=/recordings ghcr.io/miyabiver39/embeddinggemma-poc:cpu
```

`WATCH_DIRS` に指定したフォルダを定期的に走査し、新しい録画ファイルを自動で取り込みます。
書き込み中のファイルを避けるため、最終更新から `WATCH_SETTLE_SEC` 秒(既定 30 秒)経過したファイルだけを対象にします。
ファイルが入っているフォルダ名をカメラ ID にします(`/recordings/cam01/...` なら `cam01`)。状態は WebUI の「状態」タブで確認できます。

## イメージの種類

| タグ | 用途 | 起動オプション |
|---|---|---|
| `:cpu`(= `:latest`) | CPU 推論。どこでも動く | なし |
| `:cuda` | NVIDIA GPU(RTX 3060 など) | `--gpus all` |
| `:rocm` | AMD GPU(RX 9060 XT など) | `--device=/dev/kfd --device=/dev/dri --group-add video` |
| `:intel` | Intel GPU(Arc / Core Ultra 内蔵) | `--device=/dev/dri` |
| `:slim` | モデル無し。ベクトル化を外部 compute に任せる app 専用 | なし |

```bash
# NVIDIA
docker run -d -p 8000:8000 -v ./data:/data --gpus all ghcr.io/miyabiver39/embeddinggemma-poc:cuda
# AMD
docker run -d -p 8000:8000 -v ./data:/data --device=/dev/kfd --device=/dev/dri --group-add video \
  ghcr.io/miyabiver39/embeddinggemma-poc:rocm
# Intel
docker run -d -p 8000:8000 -v ./data:/data --device=/dev/dri ghcr.io/miyabiver39/embeddinggemma-poc:intel
```

GPU が使われているかは WebUI の「状態」タブ、または `curl localhost:8000/api/info` の `embedder.accelerator` で確認できます
(`nvidia-cuda` / `amd-rocm` / `intel-xpu` / `cpu`)。
**GPU 版は作者の環境(GPU なし)で動作確認できていません。** 実機での確認手順と既知の注意点は [docs/gpu.md](docs/gpu.md) を見てください。
Intel の古い内蔵 GPU(Iris Xe など)は PyTorch の XPU 対応外の可能性が高く、その場合は CPU で動きます。

## 3つの動かし方(ROLE)

| ROLE | 中身 | 使いどころ |
|---|---|---|
| `all`(既定) | 取り込み + 検索 + DB + WebUI + ベクトル化 | 1コンテナで完結させたいとき |
| `app` | 取り込み + 検索 + DB + WebUI(モデル無し) | ベクトル化を別マシンの GPU に任せたいとき |
| `compute` | ベクトル化 API だけ(`/compute/*`) | GPU サーバに置く |

```bash
# GPU サーバ側
docker run -d -p 8001:8000 --gpus all -e ROLE=compute ghcr.io/miyabiver39/embeddinggemma-poc:cuda
# 軽いマシン側
docker run -d -p 8000:8000 -v ./data:/data -e EMBEDDING_URL=http://gpu-server:8001 \
  ghcr.io/miyabiver39/embeddinggemma-poc:slim
```

`docker compose --profile split up` で同じ構成を試せます([docker-compose.yml](docker-compose.yml))。
動画の加工(フレーム抽出・音声の16kHz化)は常に app 側で行うため、compute には**画像と PCM 音声だけ**が流れます。
自前で加工済みのフレームを `/api/ingest/frames` に渡すこともできます。

## 設定(環境変数)

| 変数 | 既定 | 説明 |
|---|---|---|
| `ROLE` | `all`(slim は `app`) | `all` / `app` / `compute` |
| `EMBEDDING_BACKEND` | `local`(app は `remote`) | `local` 自プロセス / `remote` 外部 compute / `dummy` 動作確認用 |
| `EMBEDDING_URL` | `http://localhost:8001` | remote のときの compute の URL |
| `DEVICE` | `auto` | `auto` / `cpu` / `cuda`(AMD ROCm も `cuda`) / `xpu` |
| `DTYPE` | `auto` | GPU は bf16、CPU は fp32。出力が NaN になる場合は `float32` に |
| `DIMS` | `768` | 保存するベクトルの次元(768/512/256/128)。**DB 作成後は変更不可**(下記) |
| `WINDOW_SEC` | `2` | 1つのベクトルにする時間窓(秒) |
| `FRAMES_PER_WINDOW` | `2` | 窓あたりのフレーム数 |
| `OVERLAP_SEC` | `0` | 窓の重なり(秒) |
| `INCLUDE_AUDIO` | `false` | 動画の音声も同じベクトルに含める |
| `AUDIO_CHUNK_SEC` | `10` | 音声ファイルの分割長(秒) |
| `IMAGE_MAX_SIDE` | `448` | 画像を縮小する長辺(px) |
| `IMAGE_MAX_TOKENS` | `0` | 画像1枚あたりのトークン上限(0=モデル既定) |
| `TOP_K` | `10` | 検索の既定件数 |
| `DATA_DIR` | `/data` | DB・サムネイル・アップロード動画の保存先 |
| `WATCH_DIRS` | (空) | 自動で取り込む監視フォルダ(カンマ区切り)。空なら無効 |
| `WATCH_INTERVAL_SEC` | `60` | 監視フォルダを走査する間隔(秒) |
| `WATCH_SETTLE_SEC` | `30` | 最終更新からこの秒数が経過したファイルだけを取り込む(書き込み中のファイルを避ける) |
| `WATCH_CAMERA_FROM_DIR` | `true` | ファイルが入っているフォルダ名をカメラ ID にする |
| `WATCH_PRESET` | (空) | 監視フォルダの取り込みに使うプリセット(`object` / `action` / `speech`)。空なら上記の窓の既定値 |
| `API_TOKEN` | (空) | 設定すると `/api` と `/compute` にトークンが必要になる(`docs/security.md`) |
| `EMBEDDING_TOKEN` | `API_TOKEN` と同じ | remote のとき、compute に送るトークン |
| `INGEST_ROOTS` | `/recordings` | パス指定・フォルダ一括で取り込めるフォルダ(カンマ区切り)。`DATA_DIR` と監視フォルダは自動で追加 |
| `MAX_UPLOAD_MB` | `4096` | 1 リクエストの大きさの上限(MB)。`0` で無制限 |
| `ALLOWED_ORIGINS` | (空) | 別のオリジンのページから更新系の API を呼ぶ場合に許可するオリジン(カンマ区切り) |
| `ALLOWED_HOSTS` | `*` | 受け付ける Host ヘッダー(カンマ区切り)。DNS リバインディング対策 |
| `PUID` / `PGID` | `1000` | アプリを動かすユーザー / グループの ID。`PUID=0` で root のまま動かす |
| `TZ` | `Asia/Tokyo` | 時刻(ファイル名の日時、タイムゾーンのない `start_ts`)の解釈 |
| `LOG_LEVEL` | `INFO` | |

取り込みのプリセット(`preset`): `object`(2秒窓・2フレーム・音声なし。物体やシーン向け)/ `action`(4秒窓・4フレーム・音声なし。動作向け)/ `speech`(6秒窓・6フレーム・音声あり。会話や音が重要なとき向け)。

## 知っておくべき設計上のルール

- **メタデータ(カメラ・場所・時刻)はベクトルに埋め込みません。** 別の列に保存し、検索時に絞り込みます(ベクトルに混ぜると検索精度と再現性が落ちるため)。
- DB には**モデルID・次元・窓の設定**を記録し、**違う設定での追記は 409 で拒否**します(ベクトル空間の混在防止)。`DIMS` を変えたいときは `DATA_DIR` を作り直して再取り込みしてください。
- 「映像のみ(frames)」「映像+音声(tav)」「音声のみ(audio)」は**別のベクトル空間**として別々に検索します(`kind` で指定。`auto` は `INCLUDE_AUDIO` の設定に合わせて tav か frames を探します。音声で検索したときの `auto` は `audio` を探します)。
- 検索は SQLite に保存したベクトルを numpy の総当たり(コサイン類似度)で探します。件数が増えると検索時間とメモリが線形に増えます(256次元なら100万窓で約1GB)。目安は百万窓程度まで。それ以上は専用のベクトル DB への差し替えを検討してください(設計書参照)。

## API の概要

| | |
|---|---|
| `GET /api/info` | 状態・版・既定値・プリセット・索引の件数・監視フォルダの状態 |
| `POST /api/ingest/video` / `audio` | ファイルのアップロード取り込み |
| `POST /api/ingest/path` | コンテナ内パスの取り込み |
| `POST /api/ingest/dir` | コンテナ内フォルダの一括取り込み |
| `POST /api/watch/scan` | 監視フォルダを今すぐ走査 |
| `POST /api/ingest/frames` | 加工済みフレーム(+音声)の取り込み |
| `POST /api/search/text` / `image` / `audio` | 検索(カメラ・場所・期間・種別・最小スコアで絞り込み) |
| `GET /api/jobs`, `/api/sources` ほか | ジョブ・ソースの確認、削除、再取り込み |
| `GET /api/media/{id}`, `/api/thumb/{id}` | 元動画(Range 対応)とサムネイル |
| `/compute/*` | ベクトル化 API(ROLE=compute / all) |

詳細は `/docs`(Swagger UI)と [docs/design.md](docs/design.md)。

## ドキュメント

- [docs/design.md](docs/design.md) — 設計書(構成、窓の設計、DB、API、ネイティブ化の計画)
- [docs/operations.md](docs/operations.md) — 運用メモ(容量見積もり、チューニング、トラブルシュート)
- [docs/gpu.md](docs/gpu.md) — GPU モード(RTX 3060 / RX 9060 XT / Intel)の確認手順
- [docs/release.md](docs/release.md) — リリースの手順と、添付する SBOM・脆弱性検査結果
- [docs/security.md](docs/security.md) — セキュリティ対策と、残っているリスク
- [SECURITY.md](SECURITY.md) — 脆弱性の報告方法
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) — サードパーティのソフトウェアとライセンス
- [AGENTS.md](AGENTS.md) — AI エージェント向けの作業ガイド

## 開発

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # モデル不要のテスト用
pytest -q                        # ダミー埋め込みでの通しテスト(ffmpeg が必要)
pip install -e ".[model]"        # 実モデルを使う場合(transformers は AGENTS.md の固定コミットを推奨)
RUN_MODEL_TESTS=1 pytest -q -m model
EMBEDDING_BACKEND=dummy DATA_DIR=./data uvicorn --factory mediasearch.main:create_app --reload
```

## リリースと SBOM・脆弱性検査

`v0.2.0` のようなタグを push すると、版付きのイメージ(`:0.2.0-cpu` など)を公開し、
全イメージの **SBOM(SPDX / CycloneDX)と脆弱性検査の結果**を添付した GitHub のリリースを作成します。
main の最新イメージは毎週検査し、結果を GitHub Actions の実行結果に残します。詳細は [docs/release.md](docs/release.md) を参照してください。

## セキュリティ

- 脆弱性を見つけた場合は、公開の Issue ではなく、GitHub の非公開の報告機能でお知らせください(手順は [SECURITY.md](SECURITY.md))。
- 実施している対策と、利用者側で必要な設定は [docs/security.md](docs/security.md) を参照してください。
- 公開イメージの SBOM と脆弱性検査の結果は、各リリースに添付しています。上の「vulnerabilities」バッジは、main の最新イメージを毎週検査した結果(全イメージのうち最も多い件数)です。

## ライセンス

- 本リポジトリのソースコード: [Apache License 2.0](LICENSE)
- モデル(EmbeddingGemma 2、Google): Apache-2.0(モデルカードの記載による)。公開イメージに同梱しています。
- 公開イメージには、FFmpeg(GPL-2.0 以降)、PyTorch などのオープンソースのソフトウェアのほか、GPU 用のイメージには
  NVIDIA(CUDA)と Intel(oneAPI)の独自ライセンスのライブラリが含まれます。一覧は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。
