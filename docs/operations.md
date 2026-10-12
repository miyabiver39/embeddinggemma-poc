# 運用メモ

> 開発・検証用です。既定では認証が無効です(`API_TOKEN` で有効化。`docs/security.md`)。映像には個人情報が含まれ得ます。

## 起動と永続化

- 停止には最大で約 1 分かかることがあります(推論の途中の窓とベクトル DB の書き出しを待つため。CPU では 1 窓に 10 秒以上かかる)。`docker run` では `--stop-timeout 60` を付けてください(`docker-compose.yml` は `stop_grace_period: 60s` を設定済み)。途中で止めたファイルの取り込みは、次の起動で最初からやり直します。
- 保存先は `-v ./data:/data` で永続化します。中身: `mediasearch.db`(SQLite。設定・取り込み元・窓・ジョブ)、`vectors.usearch`(ベクトル DB)、`thumbs/`、`media/`(アップロードした動画・音声)。
- バックアップは、動かしたまま取れます(下の「バックアップと復元」)。`data/` を丸ごとコピーする場合は、停止してから行ってください。
- 停止・再起動しても未完了の取り込みジョブは再開されます。
- 外から使う場合は、Nginx で HTTPS を終端する構成(`deploy/compose.nginx.yml`)を使います。経路ごとのバッファ・待ち時間・上限の設定は `docs/nginx.md` を参照してください。

## 取り込みのコツ

- 取り込める拡張子: 映像 `.mp4 .m4v .mov .mkv .avi .ts .mts .m2ts .webm .flv .wmv .3gp`、音声 `.wav .mp3 .m4a .aac .flac .ogg .oga .opus .wma`、画像 `.jpg .jpeg .png .webp .bmp .tif .tiff`。
- 画像は 1 枚が 1 件の取り込み元になり、窓の種類は `image` です。8x8 ピクセル未満の画像は受け付けません。EXIF の向きは補正してから取り込みます。
  それ以外の拡張子や、ffprobe で読めないファイルは、受付の時点で 422 になります(ジョブは作られません)。
- 録画ファイル名に開始日時(`20260101_090000` など)が入っていれば、`start_ts` を省略できます。応答の `start_ts_from` で、何を採用したか(`request` 指定値 / `filename` ファイル名 / `now` 受付時刻)を確認できます。
- 同じファイルは重複して登録されません。設定を変えて取り込み直す場合は、WebUI の「再取り込み」か、`"force": true` を使います。
- 監視フォルダ(`WATCH_DIRS`)で取り込みに失敗したファイルは、自動では再処理しません。原因を取り除いたうえで、WebUI の「再取り込み」を使ってください。

- 録画ファイルが大きい場合は、アップロードではなく **`/recordings` をマウントして `/api/ingest/path`** を使うと速いです。
- `start_ts` を必ず付けると、検索結果に絶対時刻が出て、期間での絞り込みが効きます(タイムゾーンなしの ISO は `TZ`=Asia/Tokyo として扱います)。
- 取り込み速度は CPU だと実時間よりかなり遅くなります。常時ストリームを取り込むなら GPU か外部 compute を。
  実測(`:cpu` イメージ、4 vCPU・メモリ 15GB のクラウド環境、GPU なし、合成動画、既定の 2秒窓・2フレーム・448px):

  | 処理 | 実測 |
  |---|---|
  | 起動(コンテナ開始〜`/healthz` 応答。モデル読み込み込み) | 約 19 秒 |
  | 映像のみ(frames)1 窓 | 約 8.5 秒(9 窓で 76 秒)→ 実時間の約 0.24 倍速 |
  | 映像+音声(tav)1 窓 | 約 8.7 秒(3 窓で 26 秒) |
  | 音声のみ 1 区切り(既定 10 秒) | 約 1 秒(4 区切りで 4 秒) |
  | 文字で検索(ベクトル化込み、16 窓) | 約 0.09 秒 |
  | 画像で検索(ベクトル化込み) | 約 3.6 秒 |
  | メモリ使用量(取り込み後) | 約 3.7 GiB |

  推論は 1 件ずつ直列なので、複数ファイルを同時に投入しても速くはなりません(キューで順番に処理)。
  app + compute の分離構成(どちらも CPU)でも、1 窓の時間はほぼ同じでした(9 窓で 73 秒)。
- 窓の設計を変える(`WINDOW_SEC` など)と別のベクトル分布になります。同じ DB に混ぜたくない場合は `DATA_DIR` を分けてください。

## 監視と、容量の不足

- `GET /healthz` はプロセスが動いているか(再起動の判断)、`GET /readyz` は要求を処理できるか(DB・ベクトル DB・推論側・空き容量。だめなら 503)を返します。どちらも認証は要りません。
  ロードバランサーやコンテナの準備確認には `/readyz` を使います(分離構成で compute が止まっている間も 503 になる)。
- `DATA_DIR` の空き容量が `MIN_FREE_DISK_MB`(既定 1024 MB)を下回ると、新しい取り込み(アップロード・パス指定・フォルダ一括・監視フォルダ・ストリームの窓)を止めます。
  API は 507 を返し、監視フォルダのファイルは登録しないまま残し、ストリームは受信を続けて窓を保存しません(`state.skipped_low_disk`)。空きが戻ると、自動で再開します。
  検索・削除は続けられます。取り込み中のジョブは、窓をまとめて保存する前に確かめ、足りなければ再試行に回します。
- 推論側に接続できない・空き容量が足りない場合、取り込みのジョブは 30 秒・60 秒・120 秒……(最大 10 分)の間隔で `JOB_RETRIES` 回まで再試行します。
- 監視基盤からは `GET /metrics`(Prometheus 形式)を読み取ります(`docs/api.md` の 9.)。例えば次の条件で知らせると、止まる前に気付けます。
  `mediasearch_disk_ok == 0`(空き容量の不足)、`mediasearch_stream_up == 0`(ストリームの切断)、
  `rate(mediasearch_stream_dropped_windows_total[10m]) > 0`(推論が追いつかない)、
  `rate(mediasearch_search_rejected_total[5m]) > 0`(検索が混み合っている)、`mediasearch_alert_events{status="failed"} > 0`(通知が届かない)。
- ジョブの履歴は `JOB_HISTORY_DAYS`(既定 30 日)を過ぎたものを削除します(取り込み元ごとの最新のジョブは残る)。

## バックアップと復元

動作中に `mediasearch.db` と `vectors.usearch` をそのままコピーすると、書き込みの途中を写して壊れる・窓とベクトルが食い違う
ことがあります。次の方法で取ってください(取り込みの書き込みを数秒止め、DB とベクトルがそろった時点の複製を作ります)。

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/admin/backup   # 今すぐ取る
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/admin/backups          # 一覧
```

- `BACKUP_INTERVAL_HOURS=24` で、24 時間ごとに自動で取ります(再起動しても、最後に取った時刻から数える)。`BACKUP_KEEP`(既定 7)個を超えた古いものは削除します。
- 保存先は `BACKUP_DIR`(既定 `DATA_DIR/backups`)の `<日時>/` です。同じディスクではディスクの故障に備えられないため、
  別のディスク・NAS をマウントして指定するか、このフォルダを定期的に別の場所へ複製してください。
  中身は `mediasearch.db`・`vectors.usearch`・`manifest.json`(作成時刻・窓の数・ファイルの SHA-256)です。
- サムネイル(`thumbs/`)とアップロードした動画(`media/`)は含めません。書いた後に変更しないファイルなので、動作中のまま
  `rsync -a data/thumbs data/media <複製先>` などで複製できます(サムネイルは、元の動画が残っていれば作り直せます)。
- 監視: `GET /metrics` の `mediasearch_backup_last_timestamp_seconds`(最後に取った時刻)が古くなっていないかを確かめます。
  失敗の内容は `GET /api/info` の `backup.last_error` に出ます。保存先の空き容量が `MIN_FREE_DISK_MB` を下回る場合は 507 で取りません。

復元(サーバーを止めてから行います):

```bash
docker stop mediasearch
IMAGE=ghcr.io/miyabiver39/embeddinggemma-poc:cpu   # 動かしているイメージ
# バックアップが壊れていないかを確かめる(SHA-256 の照合)
docker run --rm -v ./data:/data --entrypoint python $IMAGE -m mediasearch.backup verify /data/backups/20261012-030000
# DATA_DIR に戻す。今のファイルは /data/backups/pre-restore-<日時>/ に移す
docker run --rm -v ./data:/data --entrypoint python $IMAGE -m mediasearch.backup restore /data/backups/20261012-030000
docker start mediasearch   # 起動時に、ファイルの所有者を一般ユーザーに整える
```

- 戻すと、バックアップの後に作った窓・取り込み元・通知ルールは無くなります。その間に作ったサムネイルは削除します
  (窓の ID はバックアップの時点の続きから振り直すため、残すと別の場面のサムネイルが表示される)。
  監視フォルダのファイルは、次の走査で取り込み直されます。ストリームは、戻した時点の登録で受信を再開します。
- 起動時に DB とベクトルを突き合わせるため、`vectors.usearch` が無いバックアップ(窓が 0 件のとき)でも起動できます。

## 保存期間

- `RETENTION_DAYS=30` のように設定すると、録画の時刻が 30 日より前の窓を 10 分ごとに削除します(ベクトル・サムネイルも削除)。
- ストリームだけ短くしたい場合は、登録時に `retention_hours` を指定します(例 `72`)。`0` を指定したストリームは削除しません。
- 状態は `GET /api/info` の `retention`(削除した窓の数、最後に実行した時刻、エラー)で確認できます。
- 古い録画を取り込むと、期間外の窓は次の確認ですぐに削除されます。過去の録画をまとめて取り込む場合は、保存期間を外してから行ってください。

## RTSP のストリーム

- 登録は `POST /api/streams`(例は [api.md](api.md))。登録は DB に残り、サーバーの再起動後に受信を再開します。
- 1 本につき ffmpeg のプロセスを 1 つ使い、接続したまま連続で復号します。GPU の復号(`FFMPEG_HWACCEL`)も使います(失敗したら次の接続から CPU)。
- 推論が追いつかない場合は窓を捨てます。`GET /api/streams` の `state.dropped_windows` が増え続ける場合は、窓を長くする(`window_sec`)、
  フレーム数を減らす、本数を減らす、GPU を使う、のいずれかで負荷を下げてください。
- 接続は TCP(`-rtsp_transport tcp`)で行います。15 秒で接続できない、または 30 秒フレームが届かない場合は、接続し直します。

確認の手順(手元に RTSP のカメラが無い場合。2026-10-09 に、この手順で受信・切断からの再接続・再起動後の再開を確認):

```bash
docker run -d --rm --name rtsp --network host bluenviron/mediamtx:latest          # RTSP サーバー(MIT)
ffmpeg -re -stream_loop -1 -i sample.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/cam1 &
curl -H 'Content-Type: application/json' -d '{"url": "rtsp://127.0.0.1:8554/cam1"}' http://localhost:8000/api/streams
```

## 処理時間の確認(ベンチマーク)

取り込みのジョブには、処理時間の内訳(`timings`)が記録されます。設定(GPU 復号、復号の省略、窓の長さ、推論のデバイスなど)を変えたときの効果は、次のどれかで確かめられます。

| 方法 | 内容 |
|---|---|
| `GET /api/jobs/{id}` | 1 件のジョブの `timings`(段階ごとの時間、窓の数、実時間比、復号の方式、推論のデバイス) |
| `GET /api/stats` | 直近の完了ジョブを、種類(video / audio / image)ごとに合計・平均したもの |
| `GET /api/metrics` | CPU(スレッド・ffmpeg ごと)・メモリ・ディスク・GPU の使用状況と、推論の稼働率・待ち件数(起動からの積算) |
| `python bench/bench.py` | 負荷をかけながら上の値を 1 秒ごとに記録し、速度・資源の使用率・ボトルネックの診断を `report.html` にまとめる(リポジトリに同梱。イメージには含めない。[bench/README.md](../bench/README.md)) |

```bash
python bench/bench.py ingest --path /recordings/sample.mp4 --repeat 3                 # 取り込みの速さ
python bench/bench.py search --concurrency 4 --during-ingest /recordings/sample.mp4  # 取り込み中の検索の速さ
python bench/compare.py bench/results/<前> bench/results/<後>                         # 設定を変えた前後を比べる
```

`timings.stages_ms` の段階: `probe`(ffprobe)/ `decode`(映像の復号の待ち)/ `audio`(音声の変換の待ち)/ `image`(画像の読み込み)/
`embed`(推論)/ `store`(DB への保存)/ `thumb`(サムネイルの保存)。
復号は推論と並行して進むため、`decode` は「推論側が次のフレームを待った時間」です。`embed` が大半なら推論が、`decode` が大半なら復号が律速です。
`realtime_factor`(実時間比)は、1 秒の処理で取り込めた映像・音声の秒数です。1 より大きければ、録画の速度より速く取り込めています。

参考(4 vCPU・GPU なし・ダミー推論、60 秒・1080p・30fps の H.264): 取り込み 2.6 秒(実時間比 約 23)。内訳の 8 割は映像の復号です。
実際のモデルでは推論が大半を占めます(CPU では 1 窓あたり数秒)。

## 検索の使い方

- 最小スコア(コサイン類似度)は絶対的な基準ではありません。実データで上位の値を見て決めます。
- 「音声あり」で取り込んだ動画は `kind=tav`、なしは `frames` です。`auto` は `INCLUDE_AUDIO` の設定に合わせてどちらか一方だけを探します(音声で検索したときの `auto` は、音声のみで取り込んだ `audio` を探します)。両方・音声のみも探したいときは `kind=all` を使います。

## トラブルシュート

| 症状 | 原因と対処 |
|---|---|
| 409 `IndexMismatch` | DB と現在の `MODEL_ID`/`DIMS` が違います。設定を戻すか、`DATA_DIR` を作り直して再取り込み |
| 503 | `EMBEDDING_URL` の compute に届いていません。URL とコンテナの疎通を確認 |
| 422 | ffmpeg が失敗(壊れたファイル・非対応コーデック・末尾を超えた時刻)。メッセージに ffmpeg の出力が含まれます |
| 検索結果が全部同じスコア/NaN | GPU の bf16 で NaN が出た可能性。`-e DTYPE=float32` を試す |
| GPU が使われない(`accelerator: cpu`) | `docs/gpu.md` を参照 |
| `:intel` が終了コード 139 で落ちる | 修正前のイメージの不具合(Intel GPU が見えないと、XPU の判定がセグメンテーション違反を起こす。`-e DEVICE=cpu` でも transformers の読み込み時に落ちる)。最新のイメージを pull し直すか、Intel GPU があるなら `--device=/dev/dri` を付ける |
| 422「映像・音声として扱えない拡張子です」 | 対応する拡張子のファイルを指定してください(上記「取り込みのコツ」) |
| 監視フォルダのファイルが取り込まれない | 「状態」タブで、走査の時刻・待機中の件数・エラーを確認。書き込み直後のファイルは `WATCH_SETTLE_SEC` 秒待ってから対象になります |
| 起動時に「DB と現在の設定が一致しません」と警告される | `MODEL_ID` / `DIMS` が DB 作成時と違います。このままでは取り込み・検索が 409 になります |
| 401「認証が必要です」 | `API_TOKEN` が設定されています。`Authorization: Bearer <トークン>` を付けるか、WebUI でトークンを入力してください |
| 403「取り込めるフォルダの外です」 | `INGEST_ROOTS` に、取り込みたいフォルダを追加してください(既定は `/recordings` と `DATA_DIR`) |
| 403「別のサイトからのリクエストは受け付けません」 | 別のオリジンのページから API を呼んでいます。そのオリジンを `ALLOWED_ORIGINS` に追加してください |
| 413「リクエストが大きすぎます」 | `MAX_UPLOAD_MB` を大きくするか、録画フォルダをマウントしてパス指定で取り込んでください |
| 録画ファイルを読めない(Permission denied) | アプリは uid 1000 で動きます。`-e PUID=<所有者の uid> -e PGID=<gid>` を指定するか、ファイルに読み取り権限を付けてください |
| 起動時に「以前の版の DB を現在の形式に移行しました」と出る | 正常です。以前の版で作った DB を自動で移行しました(データはそのまま使えます) |
| 500「DB の処理に失敗しました」 | ログと `detail` の内容を確認してください。`DATA_DIR` に別のアプリの DB を指定していないかも確認してください |
| 起動が遅い | モデル読み込みに数十秒かかります。`/healthz` が通れば準備完了 |

## ログ

標準出力に出ます(`docker logs mediasearch`)。`-e LOG_LEVEL=DEBUG` で詳細に。

`-e LOG_FORMAT=json` にすると、1 行 1 件の JSON(`time` / `level` / `logger` / `message`、要求の処理中のログには `request_id`)で出します。
この場合、uvicorn のアクセスログの代わりに、`mediasearch.access` がリクエスト ID・経路・状態コード・処理時間(`http`)を出します
(`/healthz`・`/readyz`・`/metrics` は除く)。問い合わせを受けたときは、応答の `X-Request-ID` の値でログを探します。
