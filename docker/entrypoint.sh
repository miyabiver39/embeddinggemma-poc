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
