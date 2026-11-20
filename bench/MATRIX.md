# 共通ベンチマーク(同時取り込み × 分割 × 窓 × 精度)

複数の機種で、同じ条件・同じ手順で計測し、結果を比べられるようにするための決まりです。
計測する人(人または AI エージェント)は、この文書のとおりに実行し、決まりから外れた点があれば、結果の `notes` に必ず書いてください。

対象の機種(2026-11 時点):

| 機種の名前(`<machine>`) | GPU | 環境 | イメージ |
|---|---|---|---|
| `rtx3060` | NVIDIA RTX 3060 12GB | Ubuntu(Linux ネイティブ) | `:cuda` |
| `rx9060xt` | AMD RX 9060 XT 16GB | Windows 11 + Docker Desktop(WSL2、`/dev/dxg`) | `:rocm` |

## 1. 目的

次の問いに、実測の数値とグラフで答えます。

1. **同時に取り込むと、取り込みが終わるまでの時間は変わるか。**
   - 1 本の動画をそのまま取り込む場合と、同じ動画を 2・4・8 本に分けて同時に取り込む場合で、速さが変わるか
   - 合計 8 本の動画を、1 本ずつ・2 本同時・4 本同時・8 本同時で取り込んだとき、終わるまでの時間はどう変わるか
2. **窓の大きさ**(1・2・4・6 秒)によって、速さとベクトルの数がどう変わるか
3. **精度を下げると**(`DTYPE`・`IMAGE_MAX_SIDE`・`DIMS`)、どれだけ速くなり、品質がどれだけ下がるか

## 2. 用語

| 用語 | 意味 |
|---|---|
| 分割数 `k` | 合計 480 秒の映像を、何本のファイルに分けて取り込むか(1 / 2 / 4 / 8) |
| 同時数 `w` | サーバーの `INGEST_WORKERS`。同時に処理するファイルの数の上限(1 / 2 / 4 / 8) |
| 窓 `win` | `window_sec`。1 窓の枚数(`frames_per_window`)は窓の秒数と同じにする(1 秒に 1 枚)。重なりは 0、音声なし |
| セル | `(精度の条件, k, w, win)` の 1 つの組み合わせ |
| 完了時間 | セルの 1 回の計測で、**最初のジョブの `created_at` から、最後のジョブの `finished_at` まで**の秒数(サーバー側の時刻。`GET /api/jobs/{id}` の値) |

推論そのものは、`w` によらず 1 回ずつ順番に実行されます(`PriorityGate`)。`w` を増やして並ぶのは、復号・前処理・保存です。

## 3. 計測の組み合わせ

### 実験 1: 分割 × 同時数 × 窓(精度は既定)

- 精度: `DTYPE=bfloat16`、`IMAGE_MAX_SIDE=448`、`DIMS=768`(GPU の既定)
- `k` × `w` は、`k ≥ w` の 10 通りです(`k < w` は、ファイルの数より同時数が多いだけで意味がないため測りません)。

| | w=1 | w=2 | w=4 | w=8 |
|---|---|---|---|---|
| k=1(480 秒 × 1 本) | ○ | | | |
| k=2(240 秒 × 2 本) | ○ | ○ | | |
| k=4(120 秒 × 4 本) | ○ | ○ | ○ | |
| k=8(60 秒 × 8 本) | ○ | ○ | ○ | ○ |

  - k=8 の行が「8 本を 1 本ずつ・2・4・8 本同時」、対角(`k = w`)が「分けて全部同時」、w=1 の列が「分けただけの影響」です。
- 窓: 1 / 2 / 4 / 6 秒。各セルを **3 回**測り、中央値を使います。
- 計 10 × 4 × 3 = 120 回です。コンテナは `w` ごとに起動し直します(4 回)。

### 実験 2: 精度(27 通り)

- `DTYPE`: `bfloat16` / `float16` / `float32`
- `IMAGE_MAX_SIDE`: 448 / 336 / 224
- `DIMS`: 768 / 256 / 128
- 上の全組み合わせ(27 通り)ごとに、コンテナを起動し直して、次を行います。
  1. 取り込み: `k=8`、窓 2 秒、`w=1` と `w=4` を各 3 回(`w` を変えるため、条件ごとに起動は 2 回)
  2. 検索: 取り込みの後、`w=1` のコンテナで、文章の検索を並行 4・15 秒(`bench.py search` と同じ方法)を 1 回
  3. 品質: 4 章の評価セットをベクトルにし、基準との差と正解率を記録する
- 基準は `float32`・448・768 です。DIMS が小さい条件では、基準のベクトルを先頭から同じ次元に切り詰め、長さを 1 にそろえてから比べます。
- `float16` で NaN が出た場合や、起動に失敗した場合は、その条件を「失敗」として記録し、次の条件へ進みます(止めない)。

## 4. 素材

実在の映像は使いません。`bench/make_samples.py` で作る合成の素材だけを使います。

```bash
python bench/make_samples.py ./bench-media --split-set
```

| 素材 | 内容 |
|---|---|
| `split/k1/` 〜 `split/k8/` | 1080p・30fps・H.264(GOP 2 秒)・480 秒の元の動画 1 本を、`-c copy` で 1 / 2 / 4 / 8 本に分けたもの。合計の長さと中身は、どの `k` でも同じ |
| `quality/images/` | 品質の評価用の合成の画像 12 枚(色と図形。1080p) |
| `quality/queries.json` | 各画像に対応する正解の文章(日本語と英語) |

素材は、機種ごとに作り直してかまいません(ffmpeg の版によりファイルは同一になりませんが、条件は同じです)。

## 5. サーバーの起動

計測スクリプト(`bench/matrix.py`)が、条件ごとにコンテナを起動し直します。
機種ごとに違うのは、GPU を渡す引数(`--docker-args`)とイメージだけです。

| 機種 | `--image` | `--docker-args` |
|---|---|---|
| `rtx3060` | `ghcr.io/miyabiver39/embeddinggemma-poc:cuda` | `--gpus all` |
| `rx9060xt` | `ghcr.io/miyabiver39/embeddinggemma-poc:rocm` | `--device=/dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro` |

- スクリプトは、環境変数 `DTYPE` / `IMAGE_MAX_SIDE` / `DIMS` / `INGEST_WORKERS` を変えて起動し、`/healthz` が応答するまで待ちます。
- そのほかの設定は既定のままです(`EMBED_BATCH`・`TORCH_COMPILE`・`FFMPEG_HWACCEL` などを変えない)。
- `DATA_DIR` は `DIMS` ごとに分けます(DB が次元を記録して照合するため)。`TORCH_CACHE_DIR` は全条件で共通にし、compile をやり直さないようにします。
- 素材のフォルダは `/recordings` に読み取り専用でマウントします。

## 6. 守ること

1. **イメージをそろえる。** 計測の前に `docker pull` し、イメージの revision(ラベル `org.opencontainers.image.revision`)と digest を記録します。機種どうしで比べるときは、同じ revision から作られたイメージを使います。`INGEST_WORKERS` を含む版(fa80754 以降)が必要です。
2. **ほかの GPU の処理を止める。** 計測の前に、GPU を使うほかのコンテナ・アプリを止めます。止められないものがあれば `notes` に書きます。
3. **ウォームアップする。** コンテナを起動するたびに、窓の大きさごとに、測らない取り込みを 1 回行ってから計測します(初回は compile やカーネルの準備で遅くなるため)。
4. **3 回測って中央値を使う。** 外れた値も消さずに記録します。
5. **途中で止まったら、続きから再開する**(`--resume`)。最初からやり直す必要はありません。条件を変えて再開してはいけません。
6. **数値は推測で書かない。** 取れなかった値(例: WSL2 の AMD では GPU の使用率が取れない)は空のままにし、理由を `notes` に書きます。
7. 計測の途中で、計測に使うマシンの操作(ほかの重い処理)をしないでください。

## 7. 記録するもの

### 1 回ごとの記録(`results.jsonl`。1 行 1 回)

| 項目 | 内容 |
|---|---|
| `experiment` | `concurrency`(実験 1)/ `precision`(実験 2)/ `search` / `quality` |
| `dtype` / `image_max_side` / `dims` | 精度の条件 |
| `k` / `w` / `win` / `rep` | セルと、何回目か(ウォームアップは `rep = 0`) |
| `completion_sec` | 完了時間(2 章) |
| `windows` | 作られた窓の合計数 |
| `sec_per_window` / `realtime_factor` | 完了時間 ÷ 窓の数 / 480 ÷ 完了時間 |
| `jobs` | ジョブごとの `timings`(decode / embed / store などの内訳)と、`created_at`・`started_at`・`finished_at` |
| `infer_util` / `gpu_util` / `gpu_mem_max_mb` / `cpu_util` | 計測中の資源の使用状況(`/api/metrics` から。取れないものは null) |
| `qps` / `latency_ms`(検索) | 件数/秒、p50 / p95 / p99 |
| `cos_mean` / `cos_min` / `nan` / `top1_acc` / `margin`(品質) | 基準との cos 類似度、NaN の有無、文章から画像を探した 1 位の正解率、正解と 2 位の点数の差 |
| `ok` / `error` | 成功したか、失敗の内容 |

### 機器の情報(`machine.json`)

機種の名前、GPU の名前とメモリ、ドライバの版、CPU の名前とコア数、メモリ、OS、Docker の版、イメージの revision と digest、計測の日時、`notes`(決まりから外れた点、気づいたこと)。

## 8. 実行の手順

```bash
# 1. 素材を作る
python bench/make_samples.py ./bench-media --split-set

# 2. 試しに少しだけ実行して、動くことを確かめる(k=1,2・w=1,2・窓 2 秒・1 回。結果は別のフォルダ)
python bench/matrix.py --machine <machine> --image <image> --docker-args "<GPU の引数>" \
  --media ./bench-media --quick --out bench/results/matrix-quick

# 3. 本番(実験 1 と実験 2)。時間がかかるので、バックグラウンドで実行する
python bench/matrix.py --machine <machine> --image <image> --docker-args "<GPU の引数>" \
  --media ./bench-media --out bench/results/matrix-<machine>

# 止まったら、同じコマンドに --resume を付けて再開する

# 4. 集計とグラフ(コミットする結果を作る)
python bench/matrix_report.py bench/results/matrix-<machine> --out bench/matrix-results/<machine>

# 2 台分がそろったら、比較のグラフも作る
python bench/matrix_report.py bench/results/matrix-rtx3060 bench/results/matrix-rx9060xt --out bench/matrix-results/compare
```

見積もり(RTX 3060): 実験 1 が約 2 時間、実験 2 が約 3 時間です(`float32` はさらに延びる見込み)。

## 9. 結果の置き場所とコミット

- 1 回ごとの生のデータ(`bench/results/`)は、容量が大きいためコミットしません(`.gitignore` で除外済み)。
- `bench/matrix-results/<machine>/` に、次をコミットします。
  - `summary.csv` / `summary.json`(セルごとの中央値)
  - `results.jsonl`(1 回ごとの記録。資源の使用状況の時系列は含めない)
  - `machine.json`
  - グラフ(`*.svg`)と `report.html`
- 結果の要約(表と考察)は、`bench/README.md` の 5 章に機種ごとの節として追記します。`AGENTS.md` の 6 章「検証状況」には、確認できたことだけを追記します。
- コミットメッセージは日本語で、1 行目に要約を書きます(例: `docs: RX 9060 XT で同時取り込み・窓・精度の共通ベンチマークを計測`)。結果とドキュメントは 1 つのコミットにまとめてかまいません。スクリプトを直した場合は、別のコミットにします。
- main へは直接 push できません。ブランチを作り、プルリクエストで入れます。

## 10. 考察に書くこと

グラフと表から、少なくとも次に答えてください。答えられない場合は、その理由を書きます。

1. 分けて同時に取り込むと(対角 `k = w`)、1 本のまま取り込む(k=1)より何 % 速くなったか、あるいは遅くなったか
2. 8 本を 2・4・8 本同時にすると、1 本ずつより何 % 短くなったか。どこで頭打ちになったか
3. 頭打ちの原因はどこか(推論の稼働率、GPU の使用率、CPU、復号の時間の内訳から判断する)
4. 窓の大きさによって、1 秒の映像あたりの処理時間と、1 日あたりのベクトルの数・容量(1 窓約 3.2KB × DIMS/768 を目安)がどう変わるか
5. 精度の条件ごとの速さと品質。既定(bf16・448・768)と比べて、速くて品質が下がらない条件はあるか。`float16` で NaN が出たか
