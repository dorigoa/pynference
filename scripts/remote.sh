#!/usr/bin/env bash
# Sync this repo to the Ubuntu test PC and optionally run a command there.
#
#   scripts/remote.sh                          # sync only
#   scripts/remote.sh python bench/h2d.py      # sync, then run inside the venv
#   scripts/remote.sh 'cmd1 && cmd2'           # quoted string = remote shell line
#
# Env: PYI_HOST (default "ubuntu"), PYI_DIR (default "pynference", relative to
# the remote $HOME). The remote .venv/ and results/ are never deleted by sync.
set -euo pipefail

HOST="${PYI_HOST:-ubuntu}"
RDIR="${PYI_DIR:-pynference}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rsync -az --delete \
    --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' --exclude 'results/' \
    --exclude '.pytest_cache/' --exclude '/scratch/' --exclude '/docs/plans/pynference/jobs/*.out/' \
    "$ROOT/" "$HOST:$RDIR/"

[ $# -eq 0 ] && exit 0

ssh -o BatchMode=yes "$HOST" "cd '$RDIR' && { [ -f .venv/bin/activate ] && . .venv/bin/activate || true; } && $*"
