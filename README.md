# vmsembed — EmbeddingGemma 2 で動画・音声・画像・文章を横断検索する(開発用)

「赤い車が映っている場面」「サイレンが鳴っている区間」のような**言葉(または画像・音声)**で、録画映像の**該当時刻**を探すための
バックエンド基盤です。VMS(映像管理システム)への組み込みを想定した開発用で、
Google の [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) を使います。

- 取り込み(ffmpeg で時間窓に分割 → ベクトル化 → SQLite に保存)と検索を **1つのコンテナ**で提供
- 動画・音声・画像・テキストを**同じベクトル空間**で検索(テキスト→映像、画像→映像、音声→音声 など)
- 開発確認用の **WebUI** 付き(`http://localhost:8000/`)、API 仕様は `/docs`(Swagger UI)
- CPU / NVIDIA GPU / AMD GPU / Intel GPU に対応(イメージを選ぶだけ)
- ビルドは GitHub Actions が行います。利用者は **`docker pull` / `docker run` だけ**です

> ## ⚠️ 必ずお読みください(自己責任)
> - **認証・暗号化・アクセス制御は一切ありません。** 信頼できるネットワーク内のバックエンドとして使う前提です。インターネットや不特定多数が届く場所に公開しないでください。
> - 録画映像・音声には**個人情報**(顔・声・車のナンバーなど)が含まれ得ます。保存されるのは「ベクトル・サムネイル・元ファイルの参照」ですが、サムネイルや元動画の配信 API は誰でも叩けます。取り扱いは利用者の責任です。
> - 開発・検証用途です。運用での利用は自己責任でお願いします。

## クイックスタート(まずは CPU)

```bash
docker run -d --name vmsembed -p 8000:8000 -v ./data:/data \
  ghcr.io/miyabiver39/embeddinggemma-poc:cpu
```

`http://localhost:8000/` を開きます。「取り込み」タブで動画を選び、「検索」タブで日本語・英語の言葉を入力します。
(モデルを同梱しているため、起動時のダウンロードはありません。起動時のモデル読み込みに数十秒かかります。)

> pull が拒否される場合: リポジトリ管理者が GitHub の Packages 設定で、各イメージの公開範囲を **Public** にする必要があります。

### 録画ディレクトリをそのまま取り込む

```bash
docker run -d --name vmsembed -p 8000:8000 -v ./data:/data -v /path/to/recordings:/recordings:ro \
  ghcr.io/miyabiver39/embeddinggemma-poc:cpu

curl -X POST localhost:8000/api/ingest/path -H 'Content-Type: application/json' \
  -d '{"path":"/recordings/cam01/20260101_090000.mp4","camera_id":"cam01","location":"玄関","start_ts":"2026-01-01T09:00:00+09:00"}'
```

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
| `TZ` | `Asia/Tokyo` | 絶対時刻(`start_ts` 省略時など)の解釈 |
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
| `GET /api/info` | 状態・既定値・プリセット・索引の件数 |
| `POST /api/ingest/video` / `audio` | ファイルのアップロード取り込み |
| `POST /api/ingest/path` | コンテナ内パスの取り込み |
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
- [AGENTS.md](AGENTS.md) — AI エージェント向けの作業ガイド

## 開発

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # モデル不要のテスト用
pytest -q                        # ダミー埋め込みでの通しテスト(ffmpeg が必要)
pip install -e ".[model]"        # 実モデルを使う場合(transformers は AGENTS.md の固定コミットを推奨)
RUN_MODEL_TESTS=1 pytest -q -m model
EMBEDDING_BACKEND=dummy DATA_DIR=./data uvicorn --factory vmsembed.main:create_app --reload
```

## ライセンス

コードは Apache License 2.0。モデル(EmbeddingGemma 2)のライセンスは Hugging Face のモデルページを確認してください。
