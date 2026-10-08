# 設計書

対象: vmsembed(EmbeddingGemma 2 による VMS 向けマルチモーダル検索基盤・開発用)

## 1. 目的と前提

- 録画映像・音声を、言葉・画像・音声で検索し、**該当する時刻**(カメラ・場所・絶対時刻つき)を返す。
- 開発用。**セキュリティ対策は行わない**(認証・TLS・入力のサンドボックス化なし)。バックエンド専用で、信頼できる NW 内に置く。
- ビルドは GitHub Actions、利用者は `docker pull` のみ。Mac 環境は考慮しない。全て Python。

## 2. 構成

```
        ┌───────────── ROLE=all(1コンテナ) ─────────────┐
 VMS ─▶ │ app: 取り込み/検索 API + WebUI + SQLite + ffmpeg │
        │        │ Embedder インターフェース                │
        │        ├─ local : 同一プロセスで推論 ────────────┤
        └────────┼─────────────────────────────────────────┘
                 └─ remote: HTTP ─▶ ROLE=compute(推論だけ。GPU サーバなど)
```

| ROLE | 載っているもの | モデル |
|---|---|---|
| `all` | app + compute | 必要 |
| `app` | 取り込み・検索・DB・WebUI | 不要(外部 compute を使う) |
| `compute` | `/compute/*` のみ | 必要 |

- **加工は常に app 側**(ffmpeg でフレーム抽出、16kHz モノラル PCM 化)。compute に流れるのは「画像(JPEG)」「PCM(int16 LE)」「テキスト」だけ。
- app に渡す前に加工済みのデータを入れたい場合は `/api/ingest/frames`(フレーム画像 + 時刻 + 任意の音声)。
- 取り込みと検索は同じプロセス。推論は `PriorityGate` で**1回ずつ直列化**し、検索(高優先)が取り込み(低優先)を推論の合間に追い越す。

### Embedder インターフェース(`src/vmsembed/embedders/`)

`embed_texts / embed_images / embed_audio / embed_parts`。実装は `LocalEmbedder`(sentence-transformers)、`RemoteEmbedder`(HTTP)、`DummyEmbedder`(テスト用)。
`embed_parts` は「ラベル文字・音声・画像」を**順序を保ったまま 1 つのベクトルに**する(動画+音声の窓に使う)。

## 3. ベクトルの設計

- モデル: `google/embeddinggemma-2`(768 次元、MRL で 768/512/256/128 に切り詰め可能。切り詰めたあと L2 正規化し直す)。
- クエリの接頭辞: `task: search result | query: `、文書側: `title: none | text: `(sentence-transformers の `prompt_name` で付与)。
- **窓(window)= N フレーム(+ 同じ時間帯の音声)= 1 ベクトル**。Google AI Edge Gallery の実装に倣う(既定 2 秒窓・2 フレーム)。
- 音声を含む窓は、映像のみの窓と**別の空間**として扱う(`kind`): `frames` / `tav` / `audio`。検索時は `kind` で選ぶ。`auto` は、文字・画像のクエリでは `INCLUDE_AUDIO` の設定に合わせて `tav`(true)か `frames`(false)を、音声のクエリ(`/api/search/audio`)では `audio` を探す(音声だけのベクトルと映像のみの窓は比べても意味のある順位にならないため)。それ以外の組み合わせは明示するか `all` で探す。
- 窓の入力レイアウト(tav): `[時刻ラベル][音声][時刻ラベル][画像]…` を 1 メッセージとして渡す。frames のみは画像列。
- トークン予算: 画像 1 枚 ≈ 280 トークン(+ラベル)、音声 25 トークン/秒 を上限 8192 に対して事前検証(超えると 400)。
  音声のトークン率は、プロセッサ(`EmbeddingGemma2Processor`)で実測して **25/秒(40ms/トークン)** と確定した(1秒=25、10秒=250、30秒=750)。
  Gallery のコメントの 6.25/秒は誤り。画像は 448px で 256 トークン(実測)なので、280 は安全側の見積もり。
- メタデータ(カメラ ID・場所・撮影時刻)は**ベクトルに含めず**、別の列に保存して SQL/配列で絞り込む。

## 4. DB(SQLite)

| テーブル | 内容 |
|---|---|
| `meta` | `model_id`, `dims`, スキーマ版。**不一致は `IndexMismatch`(HTTP 409)で拒否** |
| `sources` | 取り込み元(動画/音声)。パス、カメラ、場所、開始絶対時刻、設定、状態。パスに索引(重複確認用) |
| `windows` | `source_id`, `start_ms`, `end_ms`, `kind`, `vector`(float32 BLOB) |
| `jobs` | 取り込みジョブ(進捗・エラー)。再起動時に未完了ジョブを再開 |

検索: kind ごとの行列を numpy にキャッシュし、`行列 @ クエリ` の総当たりでコサイン類似度(正規化済み)。
絞り込み(カメラ・場所・期間・source_id・最小スコア)は配列マスクで適用。隣接する窓は結果の `intervals` に連結して返す。

容量の目安(256 次元・float32 = 1 KB/窓):

- 2 秒窓・重なりなし: 1 カメラ 1 日 = 43,200 窓 ≈ 44 MB
- 100 カメラ × 30 日 ≈ 130 GB・約 1.3 億窓 → 総当たりは不可。専用ベクトル DB(Qdrant/pgvector 等)に `Store.search` を差し替える前提で、`Store` に閉じ込めてある。

## 5. API

`README.md` の表と `/docs`(Swagger UI)を参照。エラー: 409=索引不一致、503=compute に接続不可、422=ffmpeg 失敗・映像音声として扱えないファイル・フォルダが無い、400=入力不正。
起動時にも DB と推論の設定(モデル ID・次元)を照合し、不一致ならログに警告する(起動は止めない)。

## 6. 取り込みの流れ

1. 受付(`ingest_files.FileIntake`): 拡張子と ffprobe で検証し、同じパス・同じ種類の取り込み済みがあれば重複として既存の取り込み元を返す。
   `start_ts` が無ければファイル名の日時、それも無ければ受付時刻を録画開始時刻にする。問題なければ `sources` に登録、`jobs` に積む(HTTP は即返る)。
   アップロード・パス指定・フォルダ一括(`/api/ingest/dir`)・監視フォルダ(`watcher.FolderWatcher`)はすべてこの受付を通る。
   重複確認から登録までは排他制御し、監視フォルダの走査と API が同じファイルを同時に受け付けないようにしている。
2. ワーカースレッドが `ffprobe` で長さを取得 → `plan_windows` で窓を計画。
3. 窓ごとに ffmpeg で等間隔のフレームを抽出(`IMAGE_MAX_SIDE` に縮小)。音声ありなら同区間を PCM 化して窓内の分割点に合わせて切る。
4. `embed_parts` / `embed_images` でベクトル化 → `windows` に保存、サムネイルを `thumbs/` に保存。
5. 進捗をジョブに記録。失敗時は `error` に原因を残す(ffmpeg の標準エラー末尾を含む)。

### 監視フォルダ

- `WATCH_DIRS` を設定したときだけ動く。`WATCH_INTERVAL_SEC` ごとに走査し、最終更新から `WATCH_SETTLE_SEC` 秒経過した未登録のファイルを受付に渡す。
- 一度登録したファイルは(失敗したものも含め)自動では再処理しない。壊れたファイルを毎回処理し直さないため。
- inotify ではなく定期走査にしている。Docker のバインドマウントや NAS(SMB / NFS)では変更通知が届かないことがあるため。

### メディア配信の範囲

`/api/media/{id}` が返すのは、受付の検証を通った映像・音声(`kind` が video / audio、状態が failed 以外)だけとする。
以前は任意のパスを登録でき、取り込みに失敗しても登録が残るため、コンテナ内の任意のファイル(`/etc/shadow` など)を返せてしまっていた。
これは認証の有無とは別の、入力検証の不足として扱い修正した。認証などのセキュリティ機能は引き続き持たない(AGENTS.md 第 2 節)。

## 7. デバイスと「ネイティブ化」

| 対象 | イメージ | PyTorch | 備考 |
|---|---|---|---|
| CPU | `cpu` | 2.14.1+cpu | fp32 |
| NVIDIA (RTX 3060 12GB 等) | `cuda` | 2.14.1+cu126 | bf16。ドライバは Docker の `--gpus all` |
| AMD (RX 9060 XT 16GB 等) | `rocm` | 2.14.1+rocm7.2 | PyTorch 上は `cuda` として見える。`/dev/kfd` と `/dev/dri` が必要 |
| Intel (Arc / Core Ultra 内蔵) | `intel` | 2.14.1+xpu | `xpu`。古い内蔵 GPU (Iris Xe) は未対応の可能性が高い |

「コンテナ提供時にネイティブに変換」は **フェーズ 2** として設計のみ(未実装):

- 初回起動時に PyTorch モデルを ONNX Runtime / OpenVINO / TensorRT 向けに変換し、`/data/native-cache/<model>/<device>/` に保存。2 回目以降はキャッシュを読む。
- 素材: `onnx-community/embeddinggemma-2-ONNX` に `model.onnx`(テキスト)、`vision_encoder.onnx`、`audio_encoder.onnx` があり、fp32/fp16/q4/q4f16/quantized の各版が公開されている(内容は確認済み、変換・精度比較は未実施)。
- 実装時の注意: 画像・音声エンコーダと LLM 本体が別ファイルなので、トークン埋め込みの合成部分を自前で再現する必要がある。**PyTorch 版との出力一致(コサイン類似度)を受け入れ条件**にすること。
- Nuitka 等による Python コードのネイティブコンパイルは優先度が低い(ボトルネックは推論側)。

## 8. 検証状況(重要)

| 項目 | 状態 |
|---|---|
| CPU 実モデルでの取り込み→検索(合成動画で「青い画面」→青) | 確認済み |
| ダミー埋め込みでの API 通しテスト、app/compute 分離 | 確認済み(pytest) |
| 動画+音声(tav)・音声のみ・画像クエリの実モデル動作 | モデル単体では確認済み。アプリ経由の実モデル通しは未実施 |
| GPU 3 種 | **未確認**(作者環境に GPU なし)。手順は docs/gpu.md |
| Docker ビルド、GitHub Actions、ghcr の公開設定 | **未確認**(最初の CI 実行が最初のビルド) |
| 実際の監視映像での検索精度 | 未評価 |
