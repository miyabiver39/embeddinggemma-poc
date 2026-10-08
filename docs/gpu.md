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

## Intel(内蔵 GPU / Arc)

- `--device=/dev/dri`。イメージには Level Zero / OpenCL ランタイムを入れてあります(Ubuntu 24.04 の `libze-intel-gpu1`, `intel-opencl-icd`)。
- PyTorch の XPU が正式に想定するのは Arc(A/B シリーズ)と Core Ultra の Arc グラフィックス世代。
  **Iris Xe など古い内蔵 GPU は対象外の可能性が高く**、その場合は `accelerator: cpu` で動きます(エラーにはしていません)。
- 確認: `python -c "import torch;print(torch.xpu.is_available())"`。
- 古い Intel 内蔵 GPU を使いたい場合の代替は OpenVINO(フェーズ 2 の「ネイティブ化」候補。未実装・未検証)。

## 複数 GPU・選択

`DEVICE=cuda` / `xpu` / `cpu` で強制できます。複数 GPU の NVIDIA は `--gpus '"device=0"'` などで Docker 側から絞ってください。

## 結果の記録欄

| GPU | イメージ | 起動 | accelerator | CPUとの類似度 | 取り込み速度 | メモ |
|---|---|---|---|---|---|---|
| RTX 3060 12GB | cuda | | | | | |
| RX 9060 XT 16GB | rocm | | | | | |
| Intel 内蔵 | intel | | | | | |
