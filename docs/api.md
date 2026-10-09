# API ガイド(開発者向け)

mediasearch の HTTP API の使い方をまとめます。全項目の定義は OpenAPI の仕様書にあります。

| 資料 | 場所 |
|---|---|
| **API リファレンス(Scalar。閲覧と試し呼び出し。推奨)** | `http://<サーバ>:8000/scalar` |
| 対話的な仕様書(Swagger UI) | `http://<サーバ>:8000/docs` |
| 読みやすい仕様書(ReDoc) | `http://<サーバ>:8000/redoc` |
| 仕様書(JSON) | `http://<サーバ>:8000/openapi.json`、リポジトリの [docs/openapi.json](openapi.json) |
| AI エージェントからの利用(MCP) | [docs/mcp.md](mcp.md) |

リポジトリの `docs/openapi.json` は、コードから自動で書き出したもので、内容が最新であることをテストで確認しています。
サーバを起動しなくても参照でき、コード生成ツールにそのまま渡せます。

## 基本

- すべて JSON か multipart/form-data で送り、JSON で返します(元ファイルとサムネイルの取得を除く)。
- 日時は **UNIX 秒** か **ISO 8601**(例 `2026-01-01T09:00:00`、`2026-01-01T09:00:00+09:00`)で指定します。
  タイムゾーンのない値は、サーバの `TZ`(既定 `Asia/Tokyo`)として扱います。応答の日時は ISO 8601(`abs_time` など)と UNIX 秒(`abs_ts` など)の両方で返します。
- 本ソフトウェアは開発・検証用途です。版が上がると API が変わることがあります。版は `GET /api/info` の `version` で確認できます。

## 認証

サーバに `API_TOKEN` が設定されている場合だけ必要です(既定では不要)。次のどちらかのヘッダーを付けます。

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/info
curl -H "X-API-Key: $TOKEN"            http://localhost:8000/api/info
```

Scalar(`/scalar`)では Authentication 欄、Swagger UI(`/docs`)では右上の Authorize にトークンを入力します。
`/healthz` は認証なしで呼べます(死活監視用)。仕様書(`/openapi.json`)と、仕様書の画面も認証なしで開けます(データは含みません)。

## Scalar(API リファレンス)について

`/scalar` は [Scalar](https://github.com/scalar/scalar)(MIT ライセンス)で表示しています。

- 画面のプログラムは、ブラウザが CDN(jsDelivr)から読み込みます。版を固定し、SRI のハッシュで改ざんされていないことを確認しています。
  ブラウザがインターネットに接続できない環境では表示できません。その場合は `docs/openapi.json` を手元のツールで開いてください。
- 既定の Scalar は、試し呼び出しを Scalar 社の中継サーバー経由で送ることや、利用状況の送信、AI 機能による外部への送信を行うことがあります。
  本アプリではこれらをすべて無効にしています。試し呼び出しはブラウザからこのサーバーへ直接送られます(Chromium で通信先を確認済み)。
- 画面左下の「Connect MCP」は、このサーバーの MCP(`/mcp`)への接続方法を案内します。
- Scalar の版を上げる場合は、`src/mediasearch/main.py` の `SCALAR_VERSION` と、ファイルの sha384(`SCALAR_SRI`)を更新します。

```bash
curl -sL https://cdn.jsdelivr.net/npm/@scalar/api-reference@<版>/dist/browser/standalone.js | openssl dgst -sha384 -binary | base64
```

## エラー

エラーは `{"detail": "..."}` の形で返します。説明は日本語です。入力の型の誤り(422)では、項目ごとの一覧になります。

| 状態コード | 意味 | 主な対処 |
|---|---|---|
| 400 | 入力が正しくない(日時の形式、kind の値など) | `detail` の説明に従って直す |
| 401 | 認証が必要 | トークンを付ける |
| 403 | 許可していないフォルダ(`INGEST_ROOTS`)、別サイトのページからのリクエスト、許可していないホスト名 | サーバの設定を確認する |
| 404 | 見つからない | ID を確認する |
| 409 | DB のモデル・次元と、サーバの設定が一致しない | サーバの設定を戻すか、DB を作り直す |
| 413 | リクエストが大きすぎる(`MAX_UPLOAD_MB`) | パス指定での取り込みを使う |
| 422 | 映像・音声・画像として扱えないファイル、または入力の型の誤り | ファイルの形式を確認する |
| 503 | 推論側(compute)に接続できない | compute の起動とトークンを確認する |

## 用語

| 用語 | 意味 |
|---|---|
| 取り込み元(source) | 登録した 1 つのファイル(動画・音声・画像)。`source_id` で識別します |
| 窓(window) | 検索の単位。動画は時間で区切った区間、音声は区切りごと、画像は 1 枚。`window_id` で識別します |
| 窓の種類(kind) | `frames`(映像のみ)、`tav`(映像と音声)、`audio`(音声のみ)、`image`(静止画) |
| グループ ID(group_id) | 利用側が自由に付ける識別子。検索の絞り込みに使います |
| 場所(location) | 利用側が自由に付ける名前。検索の絞り込みに使います |
| スコア(score) | クエリとのコサイン類似度。大きいほど近い。絶対的な基準ではないため、`min_score` は実データを見て決めます |

## 典型的な使い方

### 1. 取り込む

```bash
# 画像(複数可)をアップロード
curl -F files=@photo1.jpg -F files=@photo2.png -F group_id=album-1 http://localhost:8000/api/ingest/image

# 動画をアップロード
curl -F file=@clip.mp4 -F group_id=group-a http://localhost:8000/api/ingest/video

# コンテナ内のファイル(録画フォルダを /recordings にマウントしている場合)
curl -H 'Content-Type: application/json' -d '{"path": "/recordings/group-a/20260101_090000.mp4"}' \
  http://localhost:8000/api/ingest/path

# フォルダを一括で(フォルダ名をグループ ID にする)
curl -H 'Content-Type: application/json' -d '{"dir": "/recordings", "group_from_dir": true}' \
  http://localhost:8000/api/ingest/dir
```

受付の応答(抜粋):

```json
{"source_id": 12, "job_id": 34, "duplicate": false, "start_ts": "2026-01-01T09:00:00", "start_ts_from": "filename"}
```

`duplicate` が `true` の場合は取り込み済みで、新しいジョブは作られていません(取り込み直す場合は `"force": true`)。

### 2. 取り込みの完了を待つ

```bash
curl http://localhost:8000/api/jobs/34
# {"id": 34, "status": "running", "progress": 3, "total": 10, ...}
```

`status` が `done` になれば完了、`failed` なら `error` に理由があります。

### 3. 探す

```bash
# 文章で
curl -H 'Content-Type: application/json' \
  -d '{"query": "赤い車が通り過ぎる", "top_k": 5, "min_score": 0.3, "group_id": "group-a"}' \
  http://localhost:8000/api/search/text

# 画像で / 音声で(絞り込みの条件はフォームの項目で渡す)
curl -F file=@query.jpg -F top_k=5 http://localhost:8000/api/search/image
curl -F file=@query.wav http://localhost:8000/api/search/audio
```

応答(抜粋):

```json
{
  "results": [
    {"window_id": 101, "source_id": 12, "kind": "frames", "start_ms": 4000, "end_ms": 6000,
     "abs_time": "2026-01-01T09:00:04", "score": 0.71, "source_name": "20260101_090000.mp4",
     "media_url": "/api/media/12", "thumb_url": "/api/thumb/101", "group_id": "group-a", "location": null}
  ],
  "intervals": [{"source_id": 12, "start_ms": 4000, "end_ms": 8000, "score": 0.71, "window_ids": [101, 102]}],
  "took_ms": 1,
  "embed_ms": 95,
  "searched_kinds": "auto"
}
```

- `results` は窓ごと、`intervals` は同じ取り込み元で隣り合う窓をまとめた区間です(再生位置の目安)。
- `kind` を省略(`auto`)すると、文章・画像の検索は映像の窓と静止画を、音声の検索は音声の窓を探します。

### 4. 表示する

```bash
curl -o thumb.jpg http://localhost:8000/api/thumb/101          # サムネイル(JPEG)
curl -H 'Range: bytes=0-' -o clip.mp4 http://localhost:8000/api/media/12   # 元ファイル(Range 対応)
```

ブラウザの `<video>` では `"/api/media/12#t=4.0"` のように指定すると、該当の位置から再生できます。

`media_url` / `thumb_url` が `null` の場合は表示できません(音声の窓のサムネイル、または保存しない設定 `store_media=false` で取り込んだもの)。
保存しない設定では、検索結果は取り込み元の名前・時刻・スコアなどの情報だけになります。

### 5. 取り込みながら探す(リアルタイム検索)

取り込んだ窓は、ジョブの完了を待たずに、追加された時点で検索の対象になります。新しく増えた窓だけを知りたい場合は、次のどちらかを使います。

```bash
# (a) 通知を受け取る(Server-Sent Events)。新しい窓のうち、スコアが min_score 以上のものが届く
curl -N "http://localhost:8000/api/search/live?query=%E8%B5%A4%E3%81%84%E8%BB%8A&min_score=0.3&group_id=group-a"
# event: ready
# data: {"last_window_id": 120}
#
# event: hit
# data: {"window_id": 121, "source_id": 13, "score": 0.42, "abs_time": "2026-01-01T09:00:04", ...}

# (b) 定期的に問い合わせる。前回の応答の last_window_id を after_window_id に渡すと、その後に増えた窓だけを探す
curl -H 'Content-Type: application/json' -d '{"query": "赤い車", "after_window_id": 120}' http://localhost:8000/api/search/text
```

- (a) は、接続時にクエリを 1 回だけベクトル化し、以降は新しい窓だけを照合します(0.5 秒ごと)。`duration_sec`(既定 3600)で接続を閉じます。
  再接続のときは、最後に受け取った `window_id` を `after_window_id` に渡すと、取りこぼしなく続けられます。
- ブラウザでは `new EventSource("/api/search/live?query=...")` で受け取れます。WebUI では「新しい窓を通知」を「する」にして文章で検索します。
- `API_TOKEN` を設定している場合、EventSource はヘッダーを付けられないため、WebUI と同じく Cookie(`mediasearch_token`)で認証します。
- MCP の `search_text` にも `after_window_id` があります。

### 6. RTSP のストリームを取り込む

```bash
# 登録(すぐに受信を始める)。窓の設定・音声・保存しないモードは、ファイルの取り込みと同じ項目で指定できる
curl -H 'Content-Type: application/json' \
  -d '{"url": "rtsp://user:pass@192.168.1.10:554/stream1", "group_id": "entrance", "preset": "object"}' \
  http://localhost:8000/api/streams
# {"id": 1, "url": "rtsp://user:***@192.168.1.10:554/stream1", "source_id": 5, "state": {"status": "connecting", ...}}

curl http://localhost:8000/api/streams/1                # 状態
curl -X POST http://localhost:8000/api/streams/1/stop   # 停止(作った窓は残る)
curl -X POST http://localhost:8000/api/streams/1/start  # 再開
curl -X DELETE "http://localhost:8000/api/streams/1?delete_data=true"  # 登録と、作った窓を削除
```

- 1 つの登録が 1 つの取り込み元(`kind=stream`)になります。窓の `abs_time` は、受信した時刻(サーバーの時計)です。
- `state.status`: `connecting`(接続中)/ `running`(受信中)/ `retrying`(切断され、再接続を待っている。間隔は最大 60 秒)/ `stopped`。
- 推論が追いつかない場合は、古い窓を捨てて最新に追いつきます(`state.dropped_windows`、遅れは `state.lag_ms`)。
- 元のファイルが無いため、検索結果の `media_url` は `null` です(サムネイルは、`store_media=false` でなければ作ります)。

## クライアントの自動生成

`docs/openapi.json` から、各言語のクライアントを生成できます。

```bash
# Python(openapi-python-client)
pipx run openapi-python-client generate --path docs/openapi.json

# TypeScript など(OpenAPI Generator)
docker run --rm -v "$PWD:/local" openapitools/openapi-generator-cli generate \
  -i /local/docs/openapi.json -g typescript-fetch -o /local/client
```

## 仕様書の更新(このリポジトリを変更する場合)

API を変更したら、仕様書を書き出し直してコミットしてください。古いままだとテストが失敗します。

```bash
python scripts/export_openapi.py          # docs/openapi.json を更新
python scripts/export_openapi.py --check  # 最新かどうかだけ確かめる
```

要求・応答の型は `src/mediasearch/schemas.py` にあります。項目を足すときは `Field(description=...)` に日本語の説明を付けます。
