#!/bin/sh
# コンテナの起動スクリプト。ROLE が未指定なら、イメージの種類に合わせた既定値を使います。
set -e

: "${ROLE:=$(cat /etc/vmsembed_role)}"
export ROLE

# DEVICE の既定は auto。GPU を渡していなければ CPU で動きます
echo "[vmsembed] VARIANT=${VARIANT:-?} ROLE=${ROLE} DEVICE=${DEVICE:-auto} DATA_DIR=${DATA_DIR}"

exec uvicorn --factory vmsembed.main:create_app \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" --log-level "$(echo "${LOG_LEVEL:-info}" | tr 'A-Z' 'a-z')"
