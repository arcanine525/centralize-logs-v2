#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Pull this project source from a remote host to your local machine via SSH.
Uses tar over SSH to support excluding files (unlike plain scp).

Usage:
  ./pull_src.sh --host <HOST> --user <USER> [options]

Required:
  --host HOST            Remote host (IP/DNS)
  --user USER            Remote SSH username

Options:
  --port PORT            SSH port (default: 22)
  --key PATH             SSH private key path (optional)
  --remote-path PATH     Remote path to sync (default: /home/<USER>/src)
  --dest PATH            Local destination directory (default: ./src-remote)
  --include-venv         Include .venv (excluded by default)
  --dry-run              Show what would be copied without actually copying
  -h, --help             Show this help

Examples:
  ./pull_src.sh --host 203.0.113.10 --user hieplh --key ~/.ssh/id_rsa --dest ./src
  ./pull_src.sh --host myvm.example.com --user ubuntu --remote-path /home/ubuntu/src --dry-run
EOF
}

HOST="34.133.12.184"
USER="hieplh"
PORT="22"
KEY=""
REMOTE_PATH="/home/hieplh/src"
DEST="./src"
INCLUDE_VENV="0"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="${2:-}"; shift 2 ;;
    --user) USER="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --key) KEY="${2:-}"; shift 2 ;;
    --remote-path) REMOTE_PATH="${2:-}"; shift 2 ;;
    --dest) DEST="${2:-}"; shift 2 ;;
    --include-venv) INCLUDE_VENV="1"; shift ;;
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

# Build SSH options
SSH_OPTS="-o StrictHostKeyChecking=accept-new -p $PORT"
if [[ -n "$KEY" ]]; then
  SSH_OPTS="$SSH_OPTS -i $KEY"
fi

# Build tar exclude options (exclude virtual environments and cache files)
TAR_EXCLUDES="--exclude='.venv' --exclude='venv' --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' --exclude='.mypy_cache' --exclude='.ruff_cache' --exclude='.DS_Store' --exclude='node_modules' --exclude='.git' --exclude='*.egg-info'"

if [[ "$INCLUDE_VENV" == "1" ]]; then
  TAR_EXCLUDES="--exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' --exclude='.mypy_cache' --exclude='.ruff_cache' --exclude='.DS_Store' --exclude='node_modules' --exclude='.git' --exclude='*.egg-info'"
fi

REMOTE_SRC="${USER}@${HOST}"

echo "========================================"
echo "Pulling source from remote server"
echo "========================================"
echo "Host:        $HOST"
echo "Remote path: $REMOTE_PATH"
echo "Dest:        $DEST/"
echo "Excludes:    .venv, __pycache__, .git, node_modules, etc."
echo "========================================"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY-RUN] Would copy source files from $REMOTE_SRC:$REMOTE_PATH to $DEST/"
  echo "[DRY-RUN] Excluding: .venv, __pycache__, *.pyc, .git, node_modules"
  exit 0
fi

# Create destination directory
mkdir -p "$DEST"

# Use ssh + tar to copy files with exclusions
echo "Copying source files (excluding .venv, __pycache__, .git, etc.)..."
ssh $SSH_OPTS "$REMOTE_SRC" "cd $REMOTE_PATH && tar $TAR_EXCLUDES -cf - ." | tar -xf - -C "$DEST"

echo "========================================"
echo "Done! Source files copied to: $DEST/"
echo "========================================"

# Show what was copied
echo ""
echo "Contents of $DEST:"
ls -la "$DEST" | head -20
