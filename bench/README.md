# ベンチマーク(速度と資源の使用率)

起動中の mediasearch に負荷をかけながら、1 秒ごとに資源の使用状況を記録し、**どこがボトルネックか**、**資源を使い切れているか**を判断するためのツールです。

- Python 3.10 以上の標準ライブラリだけで動きます(追加のインストールは不要)。サーバーと別のマシンからでも実行できます。
- コンテナイメージには含めません。リポジトリを取得して、`bench/` から実行してください。
- サーバー側は `GET /api/metrics`(分離構成の推論側は `GET /compute/metrics`)で使用状況を返します。`API_TOKEN` を設定している場合は `--token` で渡します。

## 1. 計測用の素材を作る(任意)

実在の映像を使わずに、同じ条件で繰り返し計測できるよう、合成の映像・音声・画像を作れます(ffmpeg が必要)。

```bash
python bench/make_samples.py ./bench-media               # 1080p・720p の 60 秒(音声つき)、音声 60 秒、画像 50 枚
python bench/make_samples.py ./bench-media --seconds 300 # 長い動画で計測する
```

作ったフォルダを、サーバーの `/recordings` などにマウントします。

```bash
docker run --rm -p 8000:8000 --stop-timeout 60 -v ./data:/data -v ./bench-media:/recordings:ro \
  ghcr.io/miyabiver39/embeddinggemma-poc:cpu
```

## 2. 計測する

| シナリオ | 内容 | 例 |
|---|---|---|
| `ingest` | 動画・音声の取り込み(同じファイルを `--repeat` 回取り込み直し、中央値を出す) | `python bench/bench.py ingest --path /recordings/20260101_090000_1080p_60s.mp4 --repeat 3` |
| `images` | フォルダの画像の取り込み(枚/秒) | `python bench/bench.py images --dir /recordings/images` |
| `search` | 文章の検索の応答時間(p50 / p95 / p99)と処理量。`--during-ingest` で取り込み中の検索を測る | `python bench/bench.py search --concurrency 4 --duration 30 --during-ingest /recordings/...mp4` |
| `stream` | RTSP の受信(窓の作成の速さ・捨てた窓・遅れ)。終わったら登録と窓を削除する(`--keep` で残す) | `python bench/bench.py stream --rtsp rtsp://192.168.1.10/stream1 --duration 120` |
| `monitor` | 負荷はかけず、計測だけ行う(取り込みや検索は WebUI など別の手段で行う) | `python bench/bench.py monitor --duration 60` |

共通のオプション:

| オプション | 内容 |
|---|---|
| `--url` | サーバーの URL(既定 `http://localhost:8000`。環境変数 `MEDIASEARCH_URL` でも可) |
| `--token` | `API_TOKEN`(環境変数 `MEDIASEARCH_TOKEN` でも可) |
| `--compute-url` / `--compute-token` | 分離構成(app + compute)のとき、推論側の資源も計測する |
| `--interval` | 計測の間隔(秒。既定 1) |
| `--label` | 結果のフォルダ名に付ける名前(比較のときに分かりやすくする。例 `batch8`) |
| `--out` | 結果の保存先(既定 `bench/results/`。git の管理外) |
| `--preset` / `--window-sec` / `--frames-per-window` / `--overlap-sec` / `--audio` | 取り込みの窓の設定(`ingest` と `stream`) |

## 3. 結果を見る

`bench/results/<日時>-<シナリオ>[-<名前>]/` に保存します。

| ファイル | 内容 |
|---|---|
| `report.html` | 結果の画面(ブラウザで開く。外部への通信なし)。速度、診断、時間の推移のグラフ、CPU を使った処理の内訳、集計の表 |
| `result.json` | シナリオの結果・集計・診断・サーバーの設定 |
| `timeseries.csv` | 計測の間隔ごとの使用率(表計算ソフトで開ける) |
| `samples.jsonl` | `/api/metrics` の生の値(後から別の集計をするとき) |

設定を変えた前後の比較:

```bash
python bench/compare.py bench/results/20261012-100000-ingest-batch1 bench/results/20261012-101500-ingest-batch8
```

## 4. 計測している項目と、読み方

| 項目 | 取り方 | 読み方 |
|---|---|---|
| 推論の稼働率 | 推論が動いていた時間 ÷ 経過時間(推論は 1 件ずつ実行するため、順番待ちの制御で積算) | **100% に近ければ推論が律速**。推論を速くする(GPU・まとめ処理・窓の設定)と全体が速くなる |
| GPU 使用率・GPU メモリ | `nvidia-smi`(NVIDIA)、amdgpu の sysfs(AMD) | 推論の稼働率が高いのに GPU 使用率が低い → GPU を使い切れていない(`EMBED_BATCH` を上げる、CPU 側の前処理を確認) |
| GPU の動画復号回路の使用率 | `nvidia-smi` の decoder | 0% なら動画の復号は CPU で行われている(`FFMPEG_HWACCEL`) |
| CPU 使用率 | コンテナ全体(cgroup)の CPU 時間 ÷ 使える CPU の数 | 90% 以上なら CPU が飽和。`--cpus` の上限があれば、その数に対する割合 |
| CPU の上限で止められた時間 | cgroup の throttled | 大きければ、コンテナの CPU の上限(`--cpus`)で止められている |
| CPU の内訳 | スレッドごと(`native` = PyTorch の演算、`ingestor` = ファイルの取り込み、`stream` = RTSP、`decode` = フレームの受け取り、`worker` = API)と、ffmpeg などの子プロセス | 何が CPU を使っているか。ffmpeg が多ければ復号の負荷が高い |
| メモリ | アプリ本体の RSS、コンテナ全体(キャッシュを除く) | 上限に近ければ注意。ベクトル DB はメモリに載る(1 窓約 3.2KB) |
| ディスク | cgroup またはプロセスの読み書きの量 | 多ければディスクが律速の可能性 |
| 検索の待ち | 検索が推論の順番を待った時間 | 取り込み中は、実行中の取り込みの推論(1 窓分)を待つことがある。p99 が大きいのはこのため |
| 窓の作成・ストリームの遅れ・捨てた窓 | `/api/metrics` の index と streams | ストリームで捨てた窓があれば、推論が実時間に追いついていない |
| 処理時間の内訳 | ジョブの `timings`(probe / decode / audio / embed / store / thumb) | decode の割合が大きければ復号が律速、embed が大きければ推論が律速 |

診断(`report.html` の「診断」と、実行の最後の表示)は、これらの値から次のように判断します。

- 推論の稼働率が 85% 以上 → 推論が律速。CPU なら GPU の利用を、GPU なら使用率に応じて `EMBED_BATCH` の調整を案内
- 推論の稼働率が低く、処理時間の 30% 以上が復号の待ち → 復号が律速(`FFMPEG_HWACCEL`、`FFMPEG_SKIP_FRAMES`)
- CPU 使用率 90% 以上、CPU の上限で止められた時間 5% 以上、メモリ・GPU メモリが上限の 90% 以上、ストリームで窓を捨てた、検索の待ちが 1 秒を超えた → それぞれ注意として表示

## 5. 計測の例(GPU なし・CPU 3 コアに制限・実モデル・2026-10-12)

| シナリオ | 結果 | 診断 |
|---|---|---|
| `ingest`(20 秒の動画・2 秒窓) | 130.8 秒(1 窓 13.1 秒、実時間比 0.15) | 推論が律速(稼働率 98%)。CPU 98%(上限 3 コアで 73% の時間止められている)。CPU の内訳は PyTorch の演算 2.2 コア・取り込み 0.7 コア |
| `search --during-ingest`(並行 2) | 4.2 件/秒、p50 329 ms、p99 12.7 秒 | 取り込みの推論 1 窓分(約 13 秒)を待つことがあるため、p99 が大きい |
| `stream`(1080p の RTSP・60 秒) | 窓 7 件・捨てた窓 15 件 | 推論が実時間に追いつかない(CPU では 1 窓 2 秒以内に推論できない) |

ダミー推論(推論を省いた計測)では、60 秒・1080p の取り込みで、処理時間の 81% が復号の待ちになり「復号が律速」と判断されます。

### CPU(Ryzen 7 5700X)と GPU(RX 9060 XT 16GB / RTX 3060 12GB)の比較(2026-10-13)

素材は `make_samples.py` の既定(60 秒)。取り込みは 3 回の中央値。まとめる件数(`EMBED_BATCH`)は既定(GPU は 4)。

- CPU・RX 9060 XT: Windows 11 + Docker Desktop(WSL2)。CPU は `:cpu`(制限なし・16 スレッド)、GPU は `:rocm`(`/dev/dxg`。`docs/gpu.md`)
- RTX 3060: Ubuntu 24.04(Core i7-6700K)。`:cuda`(`--gpus all`)

| シナリオ | CPU(Ryzen 7 5700X) | RX 9060 XT(WSL2) | RTX 3060(Linux) |
|---|---|---|---|
| `ingest`(1080p・60 秒・2 秒窓 30 件) | 215.8 秒(1 窓 7.19 秒、実時間比 0.28) | 7.8 秒(1 窓 0.26 秒、実時間比 7.72) | 8.6 秒(1 窓 0.29 秒、実時間比 6.97) |
| `ingest`(音声 60 秒・10 秒区切り 6 件) | 5.17 秒(1 件 0.86 秒) | 0.91 秒(1 件 0.15 秒) | 0.99 秒(1 件 0.16 秒) |
| `images`(1080p の画像 50 枚) | 0.29 枚/秒 | 5.62 枚/秒 | 4.85 枚/秒 |
| `search`(並行 4・30 秒) | 7.1 件/秒、p50 440 ms / p95 1297 ms | 16.9 件/秒、p50 176 ms / p95 590 ms | 18.4 件/秒、p50 160 ms / p95 545 ms |

- いずれも推論が処理時間の大半を占めます(1080p の取り込みで、推論は CPU 215.0 秒・RTX 3060 7.5 秒。復号は 0.5 秒未満)。
- CPU は PyTorch が物理コア数(8)のスレッドで動くため、CPU 使用率は約 48% で頭打ちになります。
- RX 9060 XT は、高速な注意機構を有効にした状態(`rocm` イメージの既定。`docs/gpu.md`)の値です。無効のままでは 1080p の取り込みが 1 窓 1.03 秒、画像が 1.76 枚/秒でした。
- 音声と検索(短い文章)は 1 回の計算が小さいため、GPU による差が出にくくなります。
- RTX 3060 は動画の復号も GPU(NVDEC)で行います(`decoder: cuda`)。WSL2 の AMD は CPU で復号します。
- GPU はどちらも実時間より速く取り込めるため、2 秒窓の RTSP を実時間で処理できる見込みです(`stream` は未計測)。
- GPU の 1 回目の取り込みは、カーネルの準備のため遅くなることがあります(RX 9060 XT の音声で 5.3 秒。2 回目以降 1 秒弱)。

## 6. 注意

- 計測の値は、サーバーと同じマシンで他の処理が動いていると影響を受けます。比べるときは、同じ条件で複数回(`--repeat`)計測してください。
- GPU の値は、コンテナに `nvidia-smi` があるとき(NVIDIA Container Toolkit が渡す)か、AMD の GPU のときだけ取れます。Intel の GPU は使用率を取れません(PyTorch のメモリの使用量だけ)。
- `ingest` と `images` は `force` で取り込み直すため、計測に使ったファイルの窓は作り直されます。本番のデータではなく、確認用のサーバーで実行してください。
