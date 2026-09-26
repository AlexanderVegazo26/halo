#!/usr/bin/env bash
# Run a command inside WSL, log full output, print the tail and the real exit code.
#   scripts/wsl-run.sh <logname> <command...>
set -u
# L9: this script already runs arbitrary "$@" as root, so $1 is hardening-only, not a trust
# boundary -- but a $1 containing "/" or ".." would still write the log outside halo-logs.
[[ "$1" =~ ^[A-Za-z0-9_.-]+$ && "$1" != *..* ]] || { echo "wsl-run.sh: bad logname '$1'" >&2; exit 2; }
LOG="/root/halo-logs/$1.log"; shift
mkdir -p /root/halo-logs
export PATH=/root/.local/bin:/usr/local/bin:/usr/bin:/bin
"$@" >"$LOG" 2>&1
rc=$?
tail -n "${TAIL:-40}" "$LOG"
echo "EXIT=$rc (full log: $LOG)"
exit $rc
