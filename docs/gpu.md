# GPU モードの確認手順

> **作者の環境には GPU がないため、GPU 版(`cuda` / `rocm` / `intel`)は実機で動かしていません。**
> PyTorch のホイールの存在(2.14.1 の cu126 / rocm7.2 / xpu が cp312 向けに公開されていること)までは確認しましたが、
> モデルが各 GPU で正しい出力を返すかは未確認です。以下の手順で確認し、結果を Issue かコミットで残してください。

## 共通の確認(どの GPU でも)

```bash
docker run --rm -p 8000:8000 -v ./data:/data <GPUオプション> ghcr.io/miyabiver39/embeddinggemma-poc:<タグ>
curl -s localhost:8000/api/info | python3 -m json.tool | grep -A8 embedder
```

| 見る項目 | 期待値 |
|---|---|
| `accelerator` | `nvidia-cuda` / `amd-rocm` / `intel-xpu`(`cpu` なら GPU を掴めていない) |
| `dtype` | `bfloat16`(GPU)。NaN が出る場合は `-e DTYPE=float32` |

出力の正しさの確認: 同じ文章をCPU版とGPU版でベクトル化し、コサイン類似度が 0.99 以上であること
(`/compute/embed/texts` を両方に投げて比較)。bf16 のため完全一致はしません。

## NVIDIA(RTX 3060 12GB)

- ホストに NVIDIA ドライバと **NVIDIA Container Toolkit** が必要。`--gpus all` で起動。
- イメージは CUDA 12.6 ビルドの PyTorch。ドライバが古い場合は `nvidia-smi` で CUDA 12.6 以上に対応しているか確認。
- RTX 3060 は bf16 対応(Ampere)。モデルは約 740M パラメータなので 12GB に収まる想定(実測は未確認)。
- 失敗例: `could not select device driver "" with capabilities: [[gpu]]` → Container Toolkit 未導入。

## AMD(RX 9060 XT 16GB)

- ホストに ROCm 対応ドライバ(amdgpu)。`--device=/dev/kfd --device=/dev/dri --group-add video`(環境によっては `--group-add render` も)。
- イメージは ROCm 7.2 ビルドの PyTorch。**RX 9060 XT(RDNA4 / gfx1200 系)が ROCm 7.2 のホイールで有効かは未確認**。
  動かない場合は `rocm7.2` を別の版に変えて `docker/Dockerfile` を調整、または `HSA_OVERRIDE_GFX_VERSION` の利用を試す。
- PyTorch 上は `cuda` デバイスとして見える(`accelerator` は `torch.version.hip` で `amd-rocm` と判定)。
- 確認: コンテナ内で `python -c "import torch;print(torch.cuda.is_available(), torch.version.hip)"`。
- この修正より前のイメージ(2026-10-13 まで)は、`libatomic1` が入っていないため `import torch` が失敗し、推論を開始できませんでした(修正済み。
  ビルド時に `import torch` を確認するようにしています)。

### Windows(WSL2 / Docker Desktop)で AMD GPU を使う

WSL2 には `/dev/kfd` が無く、GPU は `/dev/dxg` から見えます。`rocm` イメージには、その橋渡しをする AMD の librocdxg を同梱しています。
`/dev/dxg` があり `/dev/kfd` が無い場合、起動スクリプトが必要な環境変数(`HSA_ENABLE_DXG_DETECTION=1`、
`ROCPROFILER_REGISTER_ENABLED=0`)を自動で設定します。Windows 側のドライバの一部(`libdxcore.so`)は同梱できないため、マウントで渡します。

```bash
docker run -d -p 8000:8000 -v ./data:/data --device=/dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro \
  ghcr.io/miyabiver39/embeddinggemma-poc:rocm
```

- Windows に、WSL の ROCm に対応した AMD の Adrenalin ドライバが必要です(確認した版は下の記録欄)。
- `ROCPROFILER_REGISTER_ENABLED=0` が無いと、PyTorch に同梱の rocprofiler-sdk が KFD の情報(`/sys/class/kfd`)を探して
  `Found 0 rocprofiler agents and 2 HSA agents` で異常終了します。
- **速度は Linux の NVIDIA より大きく劣ります。** RX 9060 XT(WSL2)は 1 窓約 1.03 秒で、RTX 3060(Linux)の約 0.29 秒の約 3.6 倍かかりました。
  RX 9060 XT では、GPU に小さな処理を 1 回渡すだけで約 0.1 ms かかり(1 窓の推論で数千回)、これが大半を占めると見ています。
  WSL2 の経路によるものか ROCm によるものかは、Linux ネイティブ(`/dev/kfd`)で計測するまで切り分けられていません。
- WSL2 では amdgpu の sysfs が無いため、ベンチマークの GPU 使用率は取れません(GPU メモリは PyTorch の使用量のみ)。
  また `/dev/dri` が無いため、動画の復号は CPU で行います。

## Intel(内蔵 GPU / Arc)

- `--device=/dev/dri`。イメージには Level Zero / OpenCL ランタイムを入れてあります(Ubuntu 24.04 の `libze-intel-gpu1`, `intel-opencl-icd`)。
- PyTorch の XPU が正式に想定するのは Arc(A/B シリーズ)と Core Ultra の Arc グラフィックス世代。
  **Iris Xe など古い内蔵 GPU は対象外の可能性が高く**、その場合は `accelerator: cpu` で動きます(エラーにはしていません)。
- **GPU が無い環境での既知の不具合(対処済み)**: このイメージの Intel ドライバ(Ubuntu 24.04 の `libze-intel-gpu1` 1.3.27642)と
  PyTorch XPU 版の組み合わせでは、Intel GPU が見えないと `torch.xpu.is_available()` がセグメンテーション違反で落ちます
  (`--device=/dev/dri` なしの起動で実測。ドライバを外すと正しく 0 台を返す)。アプリは判定を子プロセスで行い、落ちたら
  警告「Intel GPU(XPU)が見つからないため CPU で動きます」を出して CPU で起動します。
  実機の Intel GPU で同じ判定が通るか、また Ubuntu 24.04 のドライバが新しい世代(Arc B / Core Ultra 200 系)に対応しているかは未確認です。
  GPU があるのにこの警告が出る場合は、コンテナ内で `python -c "import torch;print(torch.xpu.device_count())"` の結果を記録してください。
- 確認: `python -c "import torch;print(torch.xpu.is_available())"`。
- 古い Intel 内蔵 GPU を使いたい場合の代替は OpenVINO(フェーズ 2 の「ネイティブ化」候補。未実装・未検証)。

## 動画の復号を GPU で行う(FFMPEG_HWACCEL)

動画の取り込みでは、ffmpeg を 1 回だけ起動して先頭から連続で復号します。GPU が使える場合は、復号も GPU で行います(推論とは別の回路を使うため、推論の速度は落ちません)。

| 設定 `FFMPEG_HWACCEL` | 動作 |
|---|---|
| `auto`(既定) | NVIDIA の GPU が渡されていれば `cuda`、`/dev/dri` があれば `vaapi`(AMD / Intel)、どちらもなければ CPU |
| `cuda` / `vaapi` / `qsv` | 指定の方式を使う |
| `none` | 常に CPU で復号する |

- GPU で復号できなかった場合(ドライバが無い、形式に対応していないなど)は、警告を出して CPU で復号し直します。取り込みは止まりません。
- 実際に使った方式は、ジョブの `timings.decoder`(`GET /api/jobs/{id}`)で確認できます。
- `vaapi` のデバイスは `/dev/dri/renderD128` などを自動で選びます。複数ある場合は `FFMPEG_HWACCEL_DEVICE` で指定します。
- イメージには、`rocm` に AMD 用(mesa)、`intel` に Intel 用(iHD)の VA-API ドライバを入れています。`cuda` は、NVIDIA Container Toolkit が
  ドライバの復号ライブラリを渡すよう、イメージに `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` を設定しています。
- **実機の GPU での復号は未確認です。** GPU が無い環境では、GPU を指定しても CPU に切り替わること(テスト `tests/test_decode.py`)までを確認しています。

実機での確認:

```bash
# 取り込んだジョブの decoder が cuda / vaapi になっていれば GPU で復号している
curl -s localhost:8000/api/jobs/1 | python3 -m json.tool | grep -A12 timings
# コンテナ内で、ffmpeg 単体で GPU の復号を試す
ffmpeg -v error -hwaccel vaapi -hwaccel_device /dev/dri/renderD128 -i <動画> -f null -   # AMD / Intel
ffmpeg -v error -hwaccel cuda -i <動画> -f null -                                       # NVIDIA
```

## 推論のまとめ処理(EMBED_BATCH)

GPU では、既定で 4 件までの窓をまとめて推論します(`EMBED_BATCH=auto`)。
RX 9060 XT(WSL2)での実測(1080p・60 秒・2 秒窓、3 回の中央値): まとめる件数 1 で 1 窓 1.05 秒、4 で 1.03 秒、8 で 0.94 秒。
GPU メモリ(PyTorch の確保量の最大)は 1 で約 3.0GB、8 で約 13.2GB でした。効果は小さく、8 は 16GB の GPU では余裕が少ないため、既定は 4 のままにしています。
RTX 3060(Linux)でも、1 で 0.32 秒、4 で 0.29 秒、8 で 0.29 秒と、効果は小さいものでした(GPU メモリは 8 でも約 3.8GB)。
4 窓をまとめて渡しても推論の時間がほぼ 4 倍になる(RX 9060 XT で実測)ため、モデル側で窓を 1 件ずつ処理している可能性があります(未確認)。
ほかの GPU では、次の手順で比べてください。

```bash
# EMBED_BATCH を変えて起動し直し、それぞれで計測して比べる(GPU 使用率・推論の稼働率・1 窓あたりの時間)
docker run ... -e EMBED_BATCH=1 ...   # まとめない
python bench/bench.py --label batch1 ingest --path /recordings/sample.mp4 --repeat 3
docker run ... -e EMBED_BATCH=8 ...
python bench/bench.py --label batch8 ingest --path /recordings/sample.mp4 --repeat 3
python bench/compare.py bench/results/*-batch1 bench/results/*-batch8
```

メモリが足りない(CUDA out of memory など)場合は、`EMBED_BATCH` を下げてください。
`report.html` の診断に「GPU を使い切れていません」と出る場合は、まとめる件数を増やす余地があります(`bench/README.md`)。

## 一般ユーザーでの実行と GPU

アプリは起動直後に一般ユーザー(uid 1000)へ切り替えて動きます(`docs/security.md`)。
`/dev/dri` と `/dev/kfd` のグループはデバイスファイルから自動で引き継ぐため、通常は追加の設定は不要です。
ただし実機の GPU では未確認です。GPU を掴めない(`accelerator: cpu`)場合は、切り分けのために `-e PUID=0`(root のまま動かす)でも試し、
結果の違いを下の記録欄に残してください。起動ログの「補助グループ=…」に、デバイスファイルのグループ番号が含まれているかも確認してください。

## 複数 GPU・選択

`DEVICE=cuda` / `xpu` / `cpu` で強制できます。複数 GPU の NVIDIA は `--gpus '"device=0"'` などで Docker 側から絞ってください。

## 結果の記録欄

| GPU | イメージ | 起動 | accelerator | CPUとの類似度 | 取り込み速度 | メモ |
|---|---|---|---|---|---|---|
| RTX 3060 12GB | cuda(Ubuntu 24.04、`--gpus all`) | 可(約 11 秒) | nvidia-cuda(bf16) | 0.9999 以上(文章 5 件・画像 3 件。NaN なし) | frames 約 0.29 秒/窓(1080p・2 秒窓・まとめる件数 4)。動画の復号も GPU(`decoder: cuda`) | ドライバ 595.99.02、NVIDIA Container Toolkit 1.20.1、`torch 2.14.1+cu126`。CPU は Core i7-6700K(2026-10-13) |
| RX 9060 XT 16GB | rocm(修正後。Windows 11 + Docker Desktop の WSL2、`/dev/dxg`) | 可(約 11 秒) | amd-rocm(bf16) | 0.9999 以上(文章 5 件・画像 3 件。NaN なし) | frames 約 1.0 秒/窓(1080p・2 秒窓・まとめる件数 4)。CPU(Ryzen 7 5700X)の約 7 倍 | ドライバ Adrenalin 26.8.1(32.0.31041.1004)、librocdxg 1.2.2、`torch 2.14.1+rocm7.2`。修正前のイメージは libatomic1 の不足で推論を開始できない |
| Intel 内蔵 | intel | | | | | |
| (GPU なし・4 vCPU) | cuda | 可(約 16 秒) | cpu | (未計測。検索スコアは cpu 版と小数 3 桁まで一致) | frames 約 8 秒/窓 | `torch 2.14.1+cu126`。GPU の確認にはならない |
| (GPU なし・4 vCPU) | intel | 修正後は可(約 15 秒) | cpu | (未計測。検索スコアは cpu 版と小数 3 桁まで一致) | frames 約 8 秒/窓 | 修正前は SIGSEGV で起動不可。GPU の確認にはならない |
