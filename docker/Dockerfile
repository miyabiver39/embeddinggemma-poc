# syntax=docker/dockerfile:1
# =============================================================================
# vmsembed イメージ。1つの Dockerfile から VARIANT で 5 種類を作ります。
#
#   VARIANT=slim   モデル無し・軽量。ROLE=app 専用(ベクトル化は外部の compute に任せる)
#   VARIANT=cpu    CPU 推論。モデル同梱。どこでも動く
#   VARIANT=cuda   NVIDIA GPU(RTX 3060 など)。モデル同梱
#   VARIANT=rocm   AMD GPU(RX 9060 XT など)。モデル同梱
#   VARIANT=intel  Intel GPU(Arc / Core Ultra の内蔵 GPU など)。モデル同梱
#
# ビルドは GitHub Actions が行います(.github/workflows/build.yml)。
# 手元でビルドする場合の例:  docker build -f docker/Dockerfile --build-arg VARIANT=cpu -t vmsembed:cpu .
#
# 注意: GPU 系(cuda / rocm / intel)は、作者の環境(GPU 無し)では動作確認できていません。
#       実機での確認手順は docs/gpu.md を参照してください。
# =============================================================================
FROM ubuntu:24.04

ARG VARIANT=cpu
# 依存のバージョンは「動作確認した組み合わせ」に固定しています(AGENTS.md 参照)
ARG TORCH_VERSION=2.14.1
ARG TORCHVISION_VERSION=0.29.1
# EmbeddingGemma 2 は transformers の正式リリース(PyPI)に未収録のため、確認済みのコミットを使う
ARG TRANSFORMERS_COMMIT=cb33194ad6152bd9fad6305378d92db385dd7b32
ARG MODEL_ID=google/embeddinggemma-2

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/hf \
    DATA_DIR=/data

# ---- OS パッケージ(ffmpeg は動画・音声の加工に必須)
#      intel のみ、Intel GPU 用の実行環境(Level Zero / OpenCL ドライバ)も入れる
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip ffmpeg ca-certificates tzdata curl \
 && if [ "$VARIANT" = "intel" ]; then \
      apt-get install -y --no-install-recommends libze1 libze-intel-gpu1 intel-opencl-icd; \
    fi \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv

# ---- PyTorch(バリアントごとに入手元が違う)。slim には入れない
RUN case "$VARIANT" in \
      slim)  echo "slim: PyTorch は入れません" ;; \
      cpu)   pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url https://download.pytorch.org/whl/cpu ;; \
      cuda)  pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url https://download.pytorch.org/whl/cu126 ;; \
      rocm)  pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url https://download.pytorch.org/whl/rocm7.2 ;; \
      intel) pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url https://download.pytorch.org/whl/xpu ;; \
      *) echo "不明な VARIANT: $VARIANT" >&2; exit 1 ;; \
    esac

# ---- アプリ本体
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN if [ "$VARIANT" = "slim" ]; then \
      pip install .; \
    else \
      pip install "https://github.com/huggingface/transformers/archive/${TRANSFORMERS_COMMIT}.tar.gz" \
      && pip install ".[model]"; \
    fi

# ---- モデルを同梱(初回起動でダウンロードしなくて済むように)。slim には入れない
RUN if [ "$VARIANT" != "slim" ]; then \
      python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}')"; \
    fi
# 同梱後は実行時にネットワークへ出ない(オフライン)
ENV HF_HUB_OFFLINE=1

# ---- 既定の動作: slim は app(外部 compute を使う)、それ以外は all(1コンテナで全部)
RUN if [ "$VARIANT" = "slim" ]; then echo app > /etc/vmsembed_role; else echo all > /etc/vmsembed_role; fi
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
 && mkdir -p /data /recordings

ENV VARIANT=${VARIANT}
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS http://localhost:${PORT:-8000}/healthz || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
