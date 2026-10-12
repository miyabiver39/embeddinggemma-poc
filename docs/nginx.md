# Nginx を前段に置く構成

信頼できるネットワークの外から使う場合は、HTTPS を終端するリバースプロキシの内側に置きます(アプリ自体は HTTP だけです)。
このリポジトリには、Nginx の設定と、Nginx とアプリを一緒に動かす compose ファイルを用意しています。

| ファイル | 内容 |
|---|---|
| `deploy/compose.nginx.yml` | Nginx(`nginx:1.28-alpine`)とアプリ。アプリのポートは外に公開せず、Nginx からだけ接続させる |
| `deploy/nginx/nginx.conf` | Nginx 本体の設定(接続数、JSON 形式のアクセスログ、圧縮、送信元ごとの上限、TLS) |
| `deploy/nginx/conf.d/mediasearch.conf` | 中継の設定(HTTP から HTTPS への転送、経路ごとのバッファ・待ち時間・上限) |
| `deploy/nginx/snippets/proxy.conf` | 中継の共通のヘッダー(Host・送信元・リクエスト ID) |
| `deploy/nginx/snippets/security-headers.conf` | Nginx で付ける応答ヘッダー(HSTS・リクエスト ID) |

## 使い方

```bash
# 1. ホスト名を書き換える(server_name、compose の ALLOWED_HOSTS・PUBLIC_URL)
sed -i 's/search.example.com/search.your-domain.example/g' deploy/nginx/conf.d/mediasearch.conf deploy/compose.nginx.yml

# 2. 証明書を置く(Let's Encrypt などで取得したもの)
mkdir -p deploy/nginx/certs
cp /path/to/fullchain.pem /path/to/privkey.pem deploy/nginx/certs/

# 3. API_TOKEN を書き換えて起動する
python3 -c "import secrets; print(secrets.token_urlsafe(32))"   # トークンを作る
docker compose -f deploy/compose.nginx.yml up -d
```

試すだけなら、自己署名の証明書でも動きます(ブラウザには警告が出ます)。

```bash
openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=search.your-domain.example" \
  -addext "subjectAltName=DNS:search.your-domain.example" \
  -keyout deploy/nginx/certs/privkey.pem -out deploy/nginx/certs/fullchain.pem
```

Let's Encrypt の証明書は、`deploy/nginx/acme` を certbot の webroot にして取得・更新できます
(`certbot certonly --webroot -w deploy/nginx/acme -d search.your-domain.example`。更新したら `docker compose -f deploy/compose.nginx.yml exec nginx nginx -s reload`)。

## 設定の要点

| 対象 | 設定 | 理由 |
|---|---|---|
| アップロード(`/api/ingest/`) | `proxy_request_buffering off`、`client_max_body_size 4096m`、待ち時間 900 秒 | 本文を Nginx の一時ファイルにためると、数 GB の動画をディスクに 2 回書き、アプリが受け取り始めるのも遅れる。上限はアプリの `MAX_UPLOAD_MB` と同じにする |
| リアルタイム検索(`/api/search/live`) | `proxy_buffering off`、待ち時間 1 時間、圧縮しない | ためると通知が遅れる。アプリは 15 秒ごとに keep-alive を送る |
| MCP(`/mcp`) | `proxy_buffering off`、待ち時間 1 時間 | 応答を SSE で返すことがある |
| 動画の配信(`/api/media/`) | `proxy_max_temp_file_size 0` | 大きいファイルを一時ファイルにためずに流す。Range(頭出し再生)はそのままアプリに渡す |
| 検索(`/api/search/`) | 送信元ごとに毎秒 5 件(瞬間的に 20 件まで)、超えたら 429 | 推論を使うため。アプリ側の全体の上限(`MAX_PENDING_SEARCHES`、503)とは別に、1 つの送信元による独占を防ぐ |
| そのほかの API | 送信元ごとに毎秒 30 件(瞬間的に 100 件まで)、同時接続 50 | 検索結果のサムネイルをまとめて読み込んでも断らない程度 |
| `/metrics` | 内部のネットワーク(127.0.0.1・10/8・172.16/12・192.168/16)からだけ | アプリ側でもトークンが必要(二重の制限) |
| `/healthz`・`/readyz` | 上限の対象外、ログを残さない | 監視からの定期的な確認を断らない |
| Host | `$http_host`(ポートを含めてそのまま) | アプリは CSRF の判定で Origin と Host を比べ、`ALLOWED_HOSTS` で Host を確かめるため |
| X-Forwarded-For | `$remote_addr`(受け取った値は引き継がない) | 送信元を偽れないようにする。アプリは `FORWARDED_ALLOW_IPS` で Nginx を信用し、ログに送信元を残す |
| X-Request-ID | 受け取った値(英数字・`-`・`_` の 64 文字まで)か、Nginx が作った値 | Nginx のログ・アプリのログ・応答で同じ値になり、問い合わせのときに突き合わせられる |
| ログ | 1 行 1 件の JSON。トークンを含むヘッダーとクエリ文字列は残さない | アプリの `LOG_FORMAT=json` と同じ形で集めやすい |
| TLS | TLS 1.2 / 1.3、HTTP/2、HSTS(1 年) | Mozilla の intermediate 相当 |

アプリ側の環境変数(`deploy/compose.nginx.yml` に記載済み):

| 環境変数 | 値 | 理由 |
|---|---|---|
| `API_TOKEN` | 推測されにくい値 | 外から使うため、必ず設定する |
| `ALLOWED_HOSTS` | `server_name` と同じ | DNS リバインディング対策(`localhost` などループバックの名前は常に許可。コンテナの死活監視のため) |
| `PUBLIC_URL` | `https://<server_name>` | 通知の Webhook の本文の URL を絶対 URL にする |
| `FORWARDED_ALLOW_IPS` | `*` | Nginx が付ける送信元のヘッダーを信用する。アプリのポートを公開せず、Nginx からだけ接続される前提のため。アプリのポートを公開する場合は Nginx の IP だけにする |
| `MAX_UPLOAD_MB` | `client_max_body_size` と同じ | どちらか小さい方で断られる |
| `LOG_FORMAT` | `json` | Nginx のログと突き合わせやすくする |

## 確認した内容(2026-10-12、nginx 1.28、ダミーの推論)

- HTTP から HTTPS への転送、HTTP/2、HSTS とリクエスト ID の付与、トークンなしは 401。
- 同じオリジンからの POST は HTTP/1.1・HTTP/2 とも通り(非標準のポートでも)、別サイトの Origin は 403。
- リアルタイム検索: 取り込みの開始から約 0.5 秒で通知が届く(Nginx でためていない)。
- 400 MB のアップロード: Nginx の一時ファイルは増えず(4 KB のまま)、そのままアプリに流れる。
- Range の要求に 206、JSON の応答は gzip、MCP の initialize が通る。
- 40 件の同時検索で 21 件が 200、19 件が 429(429 にもリクエスト ID が付く)。
- 別のコンテナから接続したとき、アプリのログの送信元は実際の IP になり、要求に付けた偽の `X-Forwarded-For` は無視される。
- `deploy/compose.nginx.yml` で、両方のコンテナが healthy になり、アップロード・検索・WebUI・`/metrics` が Nginx 経由で動く。

## 注意

- Nginx のさらに前にロードバランサーなどがある場合は、`X-Forwarded-For $remote_addr` のままだとロードバランサーの IP が送信元になります。
  Nginx の `set_real_ip_from`(realip モジュール)で、そのロードバランサーを信用する設定を追加してください。
- `/metrics` の内部のネットワークの判定は、送信元の IP で行います。Docker でポートを公開すると、ホスト自身(localhost)からの接続は
  Docker のゲートウェイ(172.x)からの接続に見えます。外からの接続は、通常は実際の IP のまま届きます。
- IPv6 で待ち受ける場合は、`listen [::]:443 ssl;` などのコメントを外します(Docker の既定のネットワークは IPv6 が無効で、そのまま有効にすると起動に失敗します)。
- 管理の操作(`/api/admin/`)を内部のネットワークに限る場合は、`/metrics` と同じ `allow` / `deny` を書きます。
