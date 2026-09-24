#!/usr/bin/env bash
# Run a command inside WSL, log full output, print the tail and the real exit code.
#   scripts/wsl-run.sh <logname> <command...>
set -u
LOG="/root/halo-logs/$1.log"; shift
mkdir -p /root/halo-logs
export PATH=/root/.local/bin:/usr/local/bin:/usr/bin:/bin
"$@" >"$LOG" 2>&1
rc=$?
tail -n "${TAIL:-40}" "$LOG"
echo "EXIT=$rc (full log: $LOG)"
exit $rc
