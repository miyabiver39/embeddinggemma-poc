# syntax=docker/dockerfile:1
# =============================================================================
# mediasearch イメージ。1つの Dockerfile から VARIANT で 5 種類を作ります。
#
#   VARIANT=slim   モデル無し・軽量。ROLE=app 専用(ベクトル化は外部の compute に任せる)
#   VARIANT=cpu    CPU 推論。モデル同梱。どこでも動く
#   VARIANT=cuda   NVIDIA GPU(RTX 3060 など)。モデル同梱
#   VARIANT=rocm   AMD GPU(RX 9060 XT など)。モデル同梱
#   VARIANT=intel  Intel GPU(Arc / Core Ultra の内蔵 GPU など)。モデル同梱
#
# ビルドは GitHub Actions が行います(.github/workflows/build.yml)。
# 手元でビルドする場合の例:  docker build -f docker/Dockerfile --build-arg VARIANT=cpu -t mediasearch:cpu .
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
# 取得したアーカイブが改ざんされていないことを、ハッシュで確かめる(pip が照合し、違えばビルドが失敗する)
ARG TRANSFORMERS_SHA256=4c30d29ff39f7398d9b0b4073594d2bcd2fbfe1bc31b12743ab08be34deef934
ARG MODEL_ID=google/embeddinggemma-2
# モデルの版(Hugging Face のコミット)を固定する。ビルドのたびに中身が変わることを防ぐ
ARG MODEL_REVISION=914f7f89142e33e77833254d9c9b90c3cef7303b

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
 && python3 -m venv /opt/venv \
 && pip install --upgrade "pip>=26.2" "setuptools>=83"
# venv に入る pip / setuptools は Ubuntu 同梱の古い版で、既知の脆弱性がある(pip-audit で確認)ため更新する

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
      pip install "transformers @ https://github.com/huggingface/transformers/archive/${TRANSFORMERS_COMMIT}.tar.gz#sha256=${TRANSFORMERS_SHA256}" \
      && pip install ".[model]"; \
    fi

# ---- モデルを同梱(初回起動でダウンロードしなくて済むように)。slim には入れない
RUN if [ "$VARIANT" != "slim" ]; then \
      python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}', revision='${MODEL_REVISION}')" \
      && chmod -R a+rX "$HF_HOME"; \
    fi
# chmod は、一般ユーザーでも読めるようにする(所有者だけが読める管理用ファイルがあるため)。同じ層で行い、モデルを複製しない
# 同梱後は実行時にネットワークへ出ない(オフライン)。読み込むモデルの版も、同梱したものに合わせる
ENV HF_HUB_OFFLINE=1 \
    MODEL_REVISION=${MODEL_REVISION}

# ---- 既定の動作: slim は app(外部 compute を使う)、それ以外は all(1コンテナで全部)
RUN if [ "$VARIANT" = "slim" ]; then echo app > /etc/mediasearch_role; else echo all > /etc/mediasearch_role; fi
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
# ---- 実行用の一般ユーザー(uid / gid 1000)。ベースイメージの ubuntu ユーザー(同じ uid)は使わないので削除する。
#      起動直後だけ root で /data の所有者などを整え、その後このユーザーに切り替える(entrypoint.sh)
#      モデルのキャッシュ(HF_HOME)は root の所有のまま読み取り専用で使う(所有者を変えると、その層にモデルが丸ごと複製されるため)
RUN chmod +x /usr/local/bin/entrypoint.sh \
 && (userdel --remove ubuntu 2>/dev/null || true) \
 && groupadd --gid 1000 mediasearch \
 && useradd --uid 1000 --gid 1000 --no-create-home --home-dir /home/mediasearch --shell /usr/sbin/nologin mediasearch \
 && mkdir -p /data /recordings /home/mediasearch \
 && chown mediasearch:mediasearch /data /home/mediasearch

ENV VARIANT=${VARIANT}
# ビルド元のコミットと参照(タグ・ブランチ)。/api/info に表示し、不具合報告で版を特定できるようにする。
# 毎回値が変わるため、ビルドキャッシュを壊さないよう最後の層で受け取る
ARG MEDIASEARCH_REVISION=unknown
ARG MEDIASEARCH_BUILD_REF=local
ENV MEDIASEARCH_REVISION=${MEDIASEARCH_REVISION} \
    MEDIASEARCH_BUILD_REF=${MEDIASEARCH_BUILD_REF}
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS http://localhost:${PORT:-8000}/healthz || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
