#!/bin/sh
# コンテナの起動スクリプト。ROLE が未指定なら、イメージの種類に合わせた既定値を使います。
#
# root で起動された場合は、データ置き場(DATA_DIR)の所有者を整えてから、一般ユーザー(PUID / PGID、既定 1000)に
# 切り替えてアプリを動かします。アプリに脆弱性があった場合の被害を、コンテナ内の root より狭くするためです。
#   - PUID=0 を指定すると、従来どおり root のまま動かします(録画ファイルが root しか読めない場合など)。
#   - docker run --user で起動した場合は、その利用者のまま動かします(所有者の調整は行いません)。
set -e

: "${ROLE:=$(cat /etc/mediasearch_role)}"
export ROLE

# DEVICE の既定は auto。GPU を渡していなければ CPU で動きます
echo "[mediasearch] VARIANT=${VARIANT:-?} ROLE=${ROLE} DEVICE=${DEVICE:-auto} DATA_DIR=${DATA_DIR}"

# AMD の GPU を Windows の WSL2(Docker Desktop)から使う場合: /dev/kfd が無く、/dev/dxg 経由になる(docs/gpu.md)。
#   - HSA_ENABLE_DXG_DETECTION=1: ROCm 7.2 の実行環境に、librocdxg(イメージに同梱)で GPU を探させる
#   - ROCPROFILER_REGISTER_ENABLED=0: PyTorch に同梱の rocprofiler-sdk が、WSL2 に無い KFD の情報(/sys/class/kfd)を読もうとして
#     プロセスごと異常終了するのを防ぐ(実測)。プロファイラを使わないため、推論には影響しない
#   - libdxcore.so は Windows 側のドライバの一部のため同梱できない。-v /usr/lib/wsl:/usr/lib/wsl:ro で渡してもらう
if [ "${VARIANT:-}" = "rocm" ] && [ -e /dev/dxg ] && [ ! -e /dev/kfd ]; then
  export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"
  export ROCPROFILER_REGISTER_ENABLED="${ROCPROFILER_REGISTER_ENABLED:-0}"
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if [ -e /usr/lib/wsl/lib/libdxcore.so ]; then
    echo "[mediasearch] WSL2 の GPU(/dev/dxg)を使います"
  else
    echo "[mediasearch] 警告: /dev/dxg はありますが /usr/lib/wsl/lib/libdxcore.so がありません。-v /usr/lib/wsl:/usr/lib/wsl:ro を付けて起動してください(docs/gpu.md)"
  fi
fi

set -- uvicorn --factory mediasearch.main:create_app \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" --log-level "$(echo "${LOG_LEVEL:-info}" | tr 'A-Z' 'a-z')"

PUID="${PUID:-1000}"
PGID="${PGID:-$PUID}"
if [ "$(id -u)" != "0" ] || [ "$PUID" = "0" ]; then
  [ "$(id -u)" = "0" ] && echo "[mediasearch] PUID=0 のため root のまま動かします"
  exec "$@"
fi

# データ置き場: 所有者が違うものだけを変更する(以前の版で root が作ったファイルの引き継ぎ。2 回目以降はほぼ何もしない)
mkdir -p "$DATA_DIR"
if ! find "$DATA_DIR" \( ! -user "$PUID" -o ! -group "$PGID" \) -exec chown "$PUID:$PGID" {} + 2>/dev/null; then
  echo "[mediasearch] 警告: ${DATA_DIR} の所有者を変更できないファイルがあります(読み取り専用のマウントなど)"
fi

# GPU のデバイスファイル(Intel / AMD の /dev/dri、AMD の /dev/kfd)のグループを引き継ぐ。
# ホストのグループ番号はコンテナ内の video / render と一致しないことがあるため、実際のファイルから調べる。
# --group-add で渡されたグループ(root のときの補助グループ)も引き継ぐ
groups=""
for g in $(id -G) $(stat -c %g /dev/dri/* /dev/kfd 2>/dev/null); do
  [ "$g" = "0" ] && continue
  case ",$groups," in *",$g,"*) ;; *) groups="${groups:+$groups,}$g" ;; esac
done

# PyTorch などが作業用のキャッシュを書き込むホーム
export HOME=/home/mediasearch
mkdir -p "$HOME" && chown "$PUID:$PGID" "$HOME"

echo "[mediasearch] 一般ユーザー(uid=${PUID} gid=${PGID}${groups:+ 補助グループ=${groups}})に切り替えて起動します"
# --no-new-privs: 切り替えた後に、setuid のプログラムなどで権限を取り戻せないようにする
if [ -n "$groups" ]; then
  exec setpriv --reuid="$PUID" --regid="$PGID" --groups="$groups" --no-new-privs -- "$@"
fi
exec setpriv --reuid="$PUID" --regid="$PGID" --clear-groups --no-new-privs -- "$@"
