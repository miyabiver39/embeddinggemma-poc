# AGENTS.md

このファイルは、AIエージェント(Claude Code など)がこのリポジトリで作業するときのガイドです。
作業を始める前に必ず読み、設計の変更があれば `docs/design.md` とあわせて更新してください。

## 1. プロジェクト概要

Google の **EmbeddingGemma 2** を使った、VMS(映像管理システム)向けのマルチモーダル検索基盤です。
テキスト・画像・動画・音声を同じベクトル空間に変換し、文章や画像で映像の該当場面を探します。

- 用途: **開発中の検証用途**です。運用品質の保証はしません。**自己責任**で利用する前提です。
- 利用形態: バックエンドとして動かします。ブラウザなど外部のフロントから直接アクセスされることは想定しません。
- WebUI: 開発用の簡易UIを同梱します(検索・取り込み・状態確認)。

## 2. 方針(変更しないこと)

| 項目 | 方針 |
|---|---|
| 実装言語 | **全てPython**(アプリ本体も推論も) |
| セキュリティ | **実施できる対策は実施する**。ただし「追加の設定なしで起動できること」は崩さない(環境に依存する対策は環境変数で有効にする方式にし、安全でない既定のまま起動したときはログで警告する)。新しい API・入力経路を足すときは、認証・CSRF・パスの制限・大きさの上限の対象になっているかを確認する。TLS はアプリでは扱わず、リバースプロキシに任せる。README に自己責任の旨と、映像が個人情報にあたり得る点を明記する。詳細は `docs/security.md` |
| ビルド | **GitHub Actions に任せる**。利用者は `docker pull` して起動するだけ |
| 公開 | 公開リポジトリ。イメージは ghcr.io へ公開する |
| 言語 | ドキュメント・コード中のコメント・docstring・WebUI は**日本語**(ビジネス文書の表現。絵文字は使わない)。識別子は英語 |
| 対象環境 | Linux コンテナ(CPU / NVIDIA CUDA / AMD ROCm / Intel XPU)。**Mac 環境は想定しない**(今後も不要) |
| 速度 | 推論は、初回起動時に ONNX Runtime / OpenVINO / TensorRT などへ変換してキャッシュできるようにする |

## 3. 構成(1つのコードベースから3種類のイメージ)

```
[app]     取り込み + 検索 + DB + ffmpeg + WebUI   (モデルなし・軽量)
[compute] ベクトル化だけ(モデル同梱。GPUはここに集約)
[all]     app + compute を1コンテナに内蔵
```

- アプリは `Embedder` インターフェースを通してベクトル化を呼びます。実装は2つです。
  - `LocalEmbedder`: 同一プロセス内で推論する(`all` 用)
  - `RemoteEmbedder`: compute コンテナの HTTP API を呼ぶ(`app` + `compute` 用)
- 切り替えは環境変数 `EMBEDDING_BACKEND=local|remote` と `EMBEDDING_URL` で行います。
- 映像・音声の加工(ffmpeg での分割・フレーム抽出・16kHzモノラル変換)は **app で行う**のが基本です。加工済みの入力を app に渡すことも許可します。
- 用途ごとに細かく API を分けます(例: `/ingest/video`, `/ingest/frames`, `/ingest/audio`, `/search/text`, `/search/image`)。
- 取り込みと検索は**同じコンテナ**で動かします。取り込み中に検索が遅くならないよう、推論の並列数を制限し、検索用の経路を空けておきます。

## 4. ベクトルとDBのルール(重要)

1. **メタデータをベクトルに埋め込まない。** カメラID・場所・時刻などは、ベクトルと並べて別のフィールドとして保存し、検索時の絞り込みに使います。
2. **モデルの混在を禁止する。** 次の値をDBに記録し、起動時・取り込み時に照合します。不一致なら取り込みを止めて警告します。
   - モデルID、精度(dtype)、次元数
   - 窓の長さ、1窓あたりのフレーム数、重なり幅、音声の有無
3. 音声ありと音声なしのベクトルは**別の空間**として扱います。設定を変えたら再インデックスが必要です。
4. MRL で次元を切り詰める場合は、DB作成時に決めて統一します(768 / 512 / 256 / 128)。
5. 検索は近い順の候補を返すだけです。**スコアの閾値**を設け、該当なしを返せるようにします。
6. 映像本体はDBに入れず、パスだけを保存します。実体は `/data` 配下に置きます。

## 5. 窓(ウィンドウ)設計

参考にした実装は Google AI Edge Gallery の Video Moments Finder です(Apache-2.0)。

- 動画は時間窓に区切り、窓ごとに複数のフレームを画像として埋め込み、**1窓=1ベクトル**にします。
- 音声を使う場合は、時刻付きで音声とフレームを交互に並べて1回で埋め込みます。
- Gallery のプリセットを初期値の目安にします(物体: 2秒・2枚・音声なし / 動作: 4秒・4枚・音声なし / 会話: 6秒・6枚・音声あり)。
- 窓が短いほどベクトル数が増えます。1台あたり1日の件数と容量を、必ず設計書に書いて確認してください。

## 6. 検証状況(実測したら更新すること)

推測で数字を書かないこと。確認したことだけを「確認済み」に書きます。

確認済み(CPU・合成動画/画像/音声):

- [x] ONNX 版リポジトリ(`onnx-community/embeddinggemma-2-ONNX`)に `model.onnx` / `vision_encoder.onnx` / `audio_encoder.onnx` があり、fp32/fp16/q4/q4f16/quantized が公開されている(変換・精度比較は未実施)
- [x] sentence-transformers 6.1.0 で text / image / audio / video(`(T,H,W,3)` 配列)/ 構造化メッセージ(順序つき混在入力)が 1 ベクトルを返す
- [x] 時刻ラベルを挟んだ交互配置(TAV)は、構造化メッセージで再現できる
- [x] `transformers` は PyPI 版に `embedding_gemma2` が無く、git のコミット `cb33194ad6152bd9fad6305378d92db385dd7b32` が必要。`torchvision` も必須
- [x] `torchcodec` はサンドボックスで読み込めない → ファイル/URL を渡さず、ffmpeg で復号した配列を渡す(設計もそうしている)
- [x] アプリ経由(実モデル・CPU)で「青い画面」→青い動画が 1 位(`RUN_MODEL_TESTS=1 pytest -m model`)
- [x] GitHub Actions でテストと 5 イメージのビルド・ghcr.io への公開が成功。匿名で `docker pull` できる(公開設定の変更は不要だった)
- [x] `:cpu` イメージを README の手順で起動 → 約 19 秒で `/healthz` 応答。WebUI(`/`)・`/docs` が開く。合成動画の取り込み → 文字検索(英語・日本語)・画像検索で正しい色の動画が 1 位
- [x] `:slim` + `:cpu`(ROLE=compute)の分離構成(`docker compose --profile split up`)で同じ通しが成功し、スコアも内蔵時と一致。compute を止めると app は起動したまま検索が 503 になる
- [x] 動画+音声(tav)・音声のみ(audio)の取り込みと、文字・音声クエリでの検索がアプリ経由(実モデル・CPU)で動く(サイン波/ノイズの区別も文字で 1 位が正しい)
- [x] 音声のトークン数は 1 秒あたり 25(40ms/トークン)。プロセッサで実測。448px の画像は 256 トークン
- [x] CPU(4 vCPU)での 1 窓あたりの処理時間: frames 約 8.5 秒、tav 約 8.7 秒、音声 10 秒区切り約 1 秒(`docs/operations.md`)
- [x] `:cuda` / `:intel` イメージは、GPU を渡さずに起動すると CPU で動き、検索スコアが `:cpu` と一致する(`:intel` は修正前はセグメンテーション違反で起動できなかった。`docs/gpu.md`)。`:rocm` は容量の都合で未実施
- [x] 取り込みの受付(拡張子・ffprobe の検証、重複確認、ファイル名の日時、フォルダ一括、監視フォルダ)が `:cpu` イメージ上で実モデルとともに動く。`/etc/shadow` などメディア以外のパスは 422 で拒否される
- [x] イメージの容量(ghcr の圧縮サイズ): slim 0.27GB / cpu 1.46GB / cuda 4.94GB / rocm 7.89GB / intel 3.82GB。展開後は slim 1.1GB / cpu 5.1GB / intel 14.1GB / cuda 14.4GB

未確認(推測を含む。実測してから記述を確定すること):

- [ ] GPU 3 種(cuda / rocm / xpu)の動作、bf16 の NaN の有無、RX 9060 XT が ROCm 7.2 ホイールで動くか
- [ ] GPU での 1 窓あたりの処理時間
- [ ] ONNX Runtime / OpenVINO / TensorRT へ変換したときのベクトル一致度と速度(フェーズ 2)
- [ ] 実際の監視映像での検索精度
- [ ] リリース時の SBOM・脆弱性検査ワークフロー(`security.yml`)のタグ契機での実行とリリースへの添付(手動実行では 5 イメージの SBOM 作成・検査・集計まで成功を確認済み)

## 7. ディレクトリ構成

```
.
├── AGENTS.md / README.md / LICENSE / docker-compose.yml
├── docs/                  design.md(設計書) operations.md(運用) gpu.md(GPU確認手順) security.md(セキュリティ対策) release.md(リリース)
├── src/vmsembed/
│   ├── main.py            アプリ生成(ROLE で構成が変わる)
│   ├── api.py             取り込み・検索・参照 API
│   ├── compute_api.py     ベクトル化 API(/compute/*)
│   ├── pipeline.py        窓の計画と取り込みワーカー
│   ├── ingest_files.py    取り込みの受付(検証・重複確認・フォルダ一括・ファイル名の日時)
│   ├── watcher.py         監視フォルダの自動取り込み(WATCH_DIRS)
│   ├── security.py        認証・CSRF・Host・本文の上限・応答ヘッダー(ASGI ミドルウェア)
│   ├── store.py           SQLite + 総当たり検索
│   ├── media.py           ffmpeg / ffprobe
│   ├── config.py          環境変数とプリセット
│   ├── embedders/         Embedder インターフェースと local / remote / dummy
│   └── web/index.html     開発用 WebUI
├── tests/                 pytest(実モデルのテストは RUN_MODEL_TESTS=1 のときだけ)
├── scripts/               release_report.py(脆弱性検査の結果をリリースノートにまとめる)
├── docker/                Dockerfile(VARIANT=slim|cpu|cuda|rocm|intel)と entrypoint
└── .github/               workflows/build.yml(テスト・ビルド・公開)、workflows/security.yml(SBOM・脆弱性検査)、dependabot.yml
```

## 7.1 デバイスの対応表

`DEVICE=auto|cpu|cuda|xpu`。ROCm 版 PyTorch でも `cuda` として見える(`torch.version.hip` で AMD と判定)。
PyTorch 2.14.1 + torchvision 0.29.1 を固定(cpu / cu126 / rocm7.2 / xpu の cp312 ホイールの存在を確認済み)。
dtype は GPU が bf16、CPU が fp32。NaN を検知したら `DTYPE=float32` を案内する。

## 8. 開発コマンド

```bash
# 依存関係のインストール(開発用)
pip install -e ".[dev]"

# テスト(ダミー埋め込み。実モデルは RUN_MODEL_TESTS=1 pytest -m model)
pytest -q

# 静的解析・整形
ruff check .

# ローカル起動(モデルを使わない動作確認用)
EMBEDDING_BACKEND=dummy DATA_DIR=./data uvicorn --factory vmsembed.main:create_app --reload
```

イメージのビルドは GitHub Actions で行います。ローカルでビルドする必要はありません。

## 9. コーディング規約

- Python 3.12 以上を想定します。型ヒントを付けます。
- コメント・docstring は日本語で、「なぜそうするか」を書きます。
- 設定は環境変数で受け、既定値は追加の設定なしで動く値にします(起動時に必須の設定を増やさない)。
- モデルの重み・映像・DBファイルを git にコミットしません(`.gitignore` で除外)。
- 外部へ通信する処理(モデルのダウンロードなど)は、失敗時に原因が分かるメッセージを出します。

## 10. コミットとPR

- コミットメッセージは日本語で、1行目に要約を書きます。
- 1つのコミットで1つの変更にします。ドキュメントだけの変更は分けます。
- コミットメッセージの末尾には、利用環境の指示(Co-Authored-By など)に従った行を付けます。
- 設計に関わる変更では、`docs/design.md` を同じコミットで更新します。
- リリースはタグ `v*` の push で作成します。SBOM と脆弱性検査の結果は自動で添付されます(`docs/release.md`)。

## 11. 作業するときの注意

- 変更前に、このファイルと `docs/design.md` を読んでください。
- 「未検証事項」にある項目は、実測の結果を書いてから、チェックを入れます。
- 方針(第2節)と矛盾する提案は、実装する前に利用者へ確認してください。
- 映像は人が映る個人情報にあたり得ます。サンプルやテストに、実在の映像を含めないでください。
