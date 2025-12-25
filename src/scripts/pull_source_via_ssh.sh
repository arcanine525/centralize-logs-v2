#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Pull this project source from a remote host to your local machine via SSH.

Usage:
  ./scripts/pull_source_via_ssh.sh --host <HOST> --user <USER> [options]

Required:
  --host HOST            Remote host (IP/DNS)
  --user USER            Remote SSH username

Options:
  --port PORT            SSH port (default: 22)
  --key PATH             SSH private key path (optional)
  --remote-path PATH     Remote path to sync (default: /home/<USER>/src)
  --dest PATH            Local destination directory (default: ./src-remote)
  --include-data         Include ./data (excluded by default)
  --include-venv         Include ./.venv (excluded by default)
  --delete               Mirror mode: delete local files not present remotely
  --dry-run              Show what would change without copying
  -h, --help             Show this help

Examples:
  ./scripts/pull_source_via_ssh.sh --host 203.0.113.10 --user hieplh --key ~/.ssh/id_rsa --dest ./centralize-logs
  ./scripts/pull_source_via_ssh.sh --host myvm.example.com --user ubuntu --remote-path /home/ubuntu/src --dry-run
EOF
}

HOST=""
USER=""
PORT="22"
KEY=""
REMOTE_PATH=""
DEST="./src-remote"
INCLUDE_DATA="0"
INCLUDE_VENV="0"
DO_DELETE="0"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="${2:-}"; shift 2 ;;
    --user) USER="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --key) KEY="${2:-}"; shift 2 ;;
    --remote-path) REMOTE_PATH="${2:-}"; shift 2 ;;
    --dest) DEST="${2:-}"; shift 2 ;;
    --include-data) INCLUDE_DATA="1"; shift ;;
    --include-venv) INCLUDE_VENV="1"; shift ;;
    --delete) DO_DELETE="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$HOST" || -z "$USER" ]]; then
  echo "Error: --host and --user are required." >&2
  usage
  exit 2
fi

if [[ -z "$REMOTE_PATH" ]]; then
  REMOTE_PATH="/home/${USER}/src"
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "Error: rsync is required on your local machine." >&2
  echo "Install it, then re-run (Ubuntu/Debian: sudo apt-get install -y rsync)." >&2
  exit 1
fi

SSH_OPTS=("-p" "$PORT" "-o" "StrictHostKeyChecking=accept-new")
if [[ -n "$KEY" ]]; then
  SSH_OPTS+=("-i" "$KEY")
fi

RSYNC_ARGS=(
  -az
  --info=stats2,progress2
  --partial
  --human-readable
)

# Excludes: keep this focused on source by default.
RSYNC_EXCLUDES=(
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.pytest_cache/'
  --exclude '.mypy_cache/'
  --exclude '.ruff_cache/'
  --exclude '.DS_Store'
)

if [[ "$INCLUDE_VENV" != "1" ]]; then
  RSYNC_EXCLUDES+=(--exclude '.venv/')
fi

if [[ "$INCLUDE_DATA" != "1" ]]; then
  RSYNC_EXCLUDES+=(--exclude 'data/')
fi

if [[ "$DO_DELETE" == "1" ]]; then
  RSYNC_ARGS+=(--delete)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

mkdir -p "$DEST"

REMOTE_SRC="${USER}@${HOST}:${REMOTE_PATH%/}/"

echo "Syncing from: $REMOTE_SRC"
echo "Syncing to:   $DEST/"

rsync "${RSYNC_ARGS[@]}" "${RSYNC_EXCLUDES[@]}" -e "ssh ${SSH_OPTS[*]}" "$REMOTE_SRC" "$DEST/"

echo "Done."
