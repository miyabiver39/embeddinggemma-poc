# MCP サーバー(AI エージェントからの利用)

mediasearch は、[Model Context Protocol(MCP)](https://modelcontextprotocol.io/) のサーバーを内蔵しています。
Claude などの AI エージェントから、自然な依頼(「先週、赤い車が映った場面を探して」など)で検索や取り込みを行えます。

| 項目 | 内容 |
|---|---|
| 接続先 | `http://<サーバ>:8000/mcp` |
| 方式 | Streamable HTTP(状態を持たない方式。JSON で応答) |
| 対象 | ROLE が `all`(既定)または `app` のコンテナ。追加の設定は不要です |
| 認証 | REST API と同じ。`API_TOKEN` を設定している場合は `Authorization: Bearer <トークン>` が必要 |

処理は REST API と同じものを使うため、検索の結果も同じです。REST API の説明は [docs/api.md](api.md) を参照してください。

## ツール

| ツール | 内容 | 種類 |
|---|---|---|
| `search_text` | 文章で探す(グループ ID・場所・期間・種類・最小スコアで絞り込み可) | 読み取り |
| `search_image` | 画像で探す(Base64 の画像、またはサーバー内のパス) | 読み取り |
| `search_audio` | 音声で探す(Base64 の音声、またはサーバー内のパス) | 読み取り |
| `get_thumbnail` | 検索結果の場面のサムネイル画像を返す(エージェントが画像で内容を確かめられる) | 読み取り |
| `ingest_path` | サーバー内のファイル(動画・音声・画像)を取り込む | 追加(同じファイルは重複しない) |
| `ingest_dir` | サーバー内のフォルダを一括で取り込む | 追加(同じファイルは重複しない) |
| `get_job` / `list_jobs` | 取り込みジョブの状態を確認する | 読み取り |
| `list_sources` | 取り込んだファイルの一覧 | 読み取り |
| `get_status` | サーバーの状態(推論のデバイス、索引の件数など) | 読み取り |

削除の操作は、誤操作を避けるため MCP には用意していません(REST API の `DELETE /api/sources/{id}` を使います)。
取り込みはサーバー内のパスで指定します(`INGEST_ROOTS` で許可したフォルダの中だけ)。

## 接続の設定

### Claude Code

```bash
# 認証なし(API_TOKEN 未設定のサーバ)
claude mcp add --transport http mediasearch http://localhost:8000/mcp

# 認証あり
claude mcp add --transport http mediasearch http://localhost:8000/mcp \
  --header "Authorization: Bearer <トークン>"
```

### 設定ファイルで指定するクライアント(例)

```json
{
  "mcpServers": {
    "mediasearch": {
      "type": "http",
      "url": "http://localhost:8000/mcp",
      "headers": { "Authorization": "Bearer <トークン>" }
    }
  }
}
```

設定ファイルの書式はクライアントごとに異なります。各クライアントの説明書に従い、「Streamable HTTP」(または「HTTP」)の接続先として上記の URL を指定してください。
標準入出力(stdio)の接続にしか対応していないクライアントは、HTTP と stdio を中継するツール(`mcp-remote` など)を使います。

### 動作の確認(curl)

```bash
curl -s http://localhost:8000/mcp \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

## 使い方の例(エージェントへの依頼)

- 「`/recordings/2026-10` のフォルダを取り込んで、終わったら教えて」
  → `ingest_dir` で受け付け、`get_job` で完了を確認します。
- 「赤い車が映っている場面を 5 件探して、それぞれの画像を見せて」
  → `search_text` で探し、`get_thumbnail` で各場面の画像を取得します。
- 「この写真に似た場面を、group-a の中から探して」
  → `search_image` に画像と `group_id` を渡します。

## 注意

- 映像・音声・画像は個人情報を含み得ます。MCP を経由すると、検索結果やサムネイルがエージェント(と、その先の AI サービス)に渡ります。
  扱ってよいデータかを確認してから接続してください。
- 信頼できるネットワークの外から使う場合は、`API_TOKEN` を設定し、TLS を終端するリバースプロキシの内側に置いてください([docs/security.md](security.md))。
- MCP の Python SDK(`mcp`、MIT ライセンス)を使っています。
