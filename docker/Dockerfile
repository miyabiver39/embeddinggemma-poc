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
# 実機で確認した GPU は、cuda(RTX 3060)と rocm(RX 9060 XT。Windows の WSL2)です。intel は未確認です。
# 確認の手順と結果は docs/gpu.md を参照してください。
# =============================================================================
# ベースイメージは、タグではなく内容のハッシュ(digest)で固定する。同じタグの中身が差し替えられても、
# 確認していないイメージでビルドしないため。更新は Dependabot が提案する(.github/dependabot.yml)
FROM ubuntu:24.04@sha256:73b999393d07f2bf65a40373778a44dc9a6d54ad7718d29247d8ff9413c4d6a8

ARG VARIANT=cpu
# 依存のバージョンは「動作確認した組み合わせ」に固定しています(AGENTS.md 参照)
ARG TORCH_VERSION=2.14.1
ARG TORCHVISION_VERSION=0.29.1
# transformers は、EmbeddingGemma 2 を収録した正式リリース(PyPI)の、動作を確認した版に固定する
# (5.23.0。以前に使っていた git のコミット cb33194 を含み、ベクトルが完全に一致することを RTX 3060 で確認。2026-11-20)
ARG TRANSFORMERS_VERSION=5.23.0
# 取得した配布物(wheel)が改ざんされていないことを、ハッシュで確かめる(pip が照合し、違えばビルドが失敗する)。
# 版を変えるときは、PyPI に載っている wheel の sha256 に合わせて更新する
ARG TRANSFORMERS_SHA256=018e82fd07490ad5bb58414f65e31576033dc7f36bed10c574843f69b90991bf
ARG MODEL_ID=google/embeddinggemma-2
# モデルの版(Hugging Face のコミット)を固定する。ビルドのたびに中身が変わることを防ぐ
ARG MODEL_REVISION=914f7f89142e33e77833254d9c9b90c3cef7303b
# cpu イメージは ONNX Runtime(fp32)を既定で同梱する(GPU の BF16 とコサイン類似度 1.000000 で完全互換、
# PyTorch FP32 より検索約 2.2 倍・動画約 1.5 倍速くメモリも少ない。PyTorch 原本 1.49 GB との二重同梱を避ける)
ARG ONNX_MODEL_ID=onnx-community/embeddinggemma-2-ONNX
ARG ONNX_MODEL_REVISION=daa72c51243991dfcaf9f9137d2c573d8f7790c0
ARG ONNX_PRECISION=fp32
# rocm のみ: Windows の WSL2(Docker Desktop)では /dev/kfd が無く、/dev/dxg 経由で GPU を使う。
# その橋渡しをする AMD の librocdxg(MIT)。RX 9060 XT で動作を確認した版に固定する(docs/gpu.md)
ARG ROCDXG_VERSION=1.2.2
ARG ROCDXG_SHA256=28ded1254811192ebace1f76c0227580184af7b27ab2475fb9728295a702d541

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
#      rocm / intel は、ffmpeg が GPU で動画を復号するための VA-API ドライバも入れる(AMD: mesa、Intel: iHD)。
#      rocm は、PyTorch の ROCm 版が読み込む libatomic1(無いと import torch が失敗する。実測)と、WSL2 用の librocdxg も入れる。
#      cuda は、NVIDIA のドライバがコンテナに渡す復号ライブラリ(NVDEC)を使う(NVIDIA_DRIVER_CAPABILITIES の video)。
#      cuda は、推論の高速化(torch.compile)のため C コンパイラと Python のヘッダーも入れる(Triton が GPU の
#      計算を呼び出す部品を、起動時にコンパイルする。無いと高速化を使わずに動く。src/mediasearch/embedders/accel.py)
#      配布元(archive.ubuntu.com)の一時的な失敗(HTTP 5xx)でビルドが止まらないよう、apt-get update を最大 5 回やり直す
#      (2026-11-20、cpu のビルドだけが InRelease の 520 で失敗した。apt の Acquire::Retries は、update の 520 を
#      やり直さないことを確かめたため、シェルで繰り返す)。Error-Mode=any で、一部の索引の失敗も失敗として扱う
RUN for i in 1 2 3 4 5; do \
      apt-get update -o APT::Update::Error-Mode=any && break; \
      if [ "$i" = 5 ]; then exit 1; fi; \
      echo "apt-get update に失敗したため、$((i * 15)) 秒後にやり直します($i/5)"; sleep $((i * 15)); \
    done \
 && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip ffmpeg ca-certificates tzdata curl \
 && if [ "$VARIANT" = "cuda" ]; then \
      apt-get install -y --no-install-recommends gcc libc6-dev python3-dev; \
    fi \
 && if [ "$VARIANT" = "intel" ]; then \
      apt-get install -y --no-install-recommends libze1 libze-intel-gpu1 intel-opencl-icd intel-media-va-driver; \
    fi \
 && if [ "$VARIANT" = "rocm" ]; then \
      apt-get install -y --no-install-recommends mesa-va-drivers libatomic1 \
      && curl -fsSLo /tmp/rocdxg.deb "https://github.com/ROCm/librocdxg/releases/download/v${ROCDXG_VERSION}/rocdxg-roct_${ROCDXG_VERSION}_amd64.deb" \
      && echo "${ROCDXG_SHA256}  /tmp/rocdxg.deb" | sha256sum -c - \
      && dpkg-deb -x /tmp/rocdxg.deb / \
      && rm /tmp/rocdxg.deb \
      && ln -sf "/opt/rocm/lib/librocdxg.so.${ROCDXG_VERSION}" /usr/lib/librocdxg.so \
      && mkdir -p /usr/share/rocdxg && cp /opt/rocm/share/rocdxg/dids.conf /usr/share/rocdxg/; \
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
    esac \
 && if [ "$VARIANT" != "slim" ]; then python -c "import torch; print('torch', torch.__version__)"; fi
# import の確認: 共有ライブラリの不足(rocm の libatomic1 など)を、公開前のビルドで検出する。GPU が無くても通る

# ---- アプリ本体
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN if [ "$VARIANT" = "slim" ]; then \
      pip install .; \
    elif [ "$VARIANT" = "cpu" ]; then \
      pip install "transformers @ https://files.pythonhosted.org/packages/py3/t/transformers/transformers-${TRANSFORMERS_VERSION}-py3-none-any.whl#sha256=${TRANSFORMERS_SHA256}" \
      && pip install ".[model,onnx]"; \
    else \
      pip install "transformers @ https://files.pythonhosted.org/packages/py3/t/transformers/transformers-${TRANSFORMERS_VERSION}-py3-none-any.whl#sha256=${TRANSFORMERS_SHA256}" \
      && pip install ".[model]"; \
    fi

# ---- モデルを同梱(初回起動でダウンロードしなくて済むように)。slim には入れない
#      cpu イメージは ONNX Runtime 用の fp32 モデルとプロセッサのみを /opt/onnx へ同梱する(二重同梱を避ける)。
#      GPU イメージ(cuda / rocm / intel)は PyTorch 用の原本モデルを HF_HOME へ同梱する。
RUN if [ "$VARIANT" = "cpu" ]; then \
      python -c "from huggingface_hub import snapshot_download; from mediasearch.embedders.onnx import _PROCESSOR_PATTERNS, required_onnx_files; files = required_onnx_files('${ONNX_PRECISION}'); snapshot_download('${ONNX_MODEL_ID}', revision='${ONNX_MODEL_REVISION}', local_dir='/opt/onnx', allow_patterns=list(_PROCESSOR_PATTERNS) + [f + '*' for f in files])" \
      && rm -rf /opt/onnx/.cache \
      && python -c "import onnxruntime; from transformers import AutoProcessor; AutoProcessor.from_pretrained('/opt/onnx', local_files_only=True); print('onnxruntime', onnxruntime.__version__)" \
      && echo /opt/onnx > /etc/mediasearch_onnx_dir \
      && chmod -R a+rX /opt/onnx; \
    elif [ "$VARIANT" != "slim" ]; then \
      python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}', revision='${MODEL_REVISION}')" \
      && chmod -R a+rX "$HF_HOME"; \
    fi
# chmod は、一般ユーザーでも読めるようにする(所有者だけが読める管理用ファイルがあるため)。同じ層で行い、モデルを複製しない
# 同梱後は実行時にネットワークへ出ない(オフライン)。読み込むモデルの版も、同梱したものに合わせる
ENV HF_HUB_OFFLINE=1 \
    MODEL_REVISION=${MODEL_REVISION} \
    ONNX_MODEL_ID=${ONNX_MODEL_ID} \
    ONNX_MODEL_REVISION=${ONNX_MODEL_REVISION} \
    ONNX_PRECISION=${ONNX_PRECISION}

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

ENV VARIANT=${VARIANT} \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
# NVIDIA_DRIVER_CAPABILITIES: 推論(compute)に加え、GPU での動画の復号(video)に必要なライブラリをコンテナに渡す。
# NVIDIA 以外の環境では使われない
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
