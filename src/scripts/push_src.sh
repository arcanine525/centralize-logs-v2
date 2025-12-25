#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Push local source to a remote host via SSH.
Uses tar over SSH to support excluding files (unlike plain scp).

Usage:
  ./push_src.sh --host <HOST> --user <USER> [options]

Required:
  --host HOST            Remote host (IP/DNS)
  --user USER            Remote SSH username

Options:
  --port PORT            SSH port (default: 22)
  --key PATH             SSH private key path (optional)
  --src PATH             Local source directory to push (default: ./src)
  --remote-path PATH     Remote destination path (default: /home/<USER>/src)
  --include-venv         Include .venv (excluded by default)
  --clean                Delete all remote contents before pushing (overwrite)
  --no-clean             Merge with existing files (don't delete)
  --dry-run              Show what would be copied without actually copying
  -h, --help             Show this help

Examples:
  ./push_src.sh --host 203.0.113.10 --user hieplh --key ~/.ssh/id_rsa
  ./push_src.sh --host myvm.example.com --user ubuntu --src ./src --remote-path /home/ubuntu/src --dry-run
EOF
}

HOST="34.133.12.184"
USER="hieplh"
PORT="22"
KEY=""
SRC="./"
REMOTE_PATH="/home/hieplh/src"
INCLUDE_VENV="0"
CLEAN="1"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="${2:-}"; shift 2 ;;
    --user) USER="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --key) KEY="${2:-}"; shift 2 ;;
    --src) SRC="${2:-}"; shift 2 ;;
    --remote-path) REMOTE_PATH="${2:-}"; shift 2 ;;
    --include-venv) INCLUDE_VENV="1"; shift ;;
    --clean) CLEAN="1"; shift ;;
    --no-clean) CLEAN="0"; shift ;;
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

# Validate source directory exists
if [[ ! -d "$SRC" ]]; then
  echo "Error: Source directory '$SRC' does not exist." >&2
  exit 1
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

REMOTE_DEST="${USER}@${HOST}"

echo "========================================"
echo "Pushing source to remote server"
echo "========================================"
echo "Host:        $HOST"
echo "Source:      $SRC/"
echo "Remote path: $REMOTE_PATH"
echo "Clean mode:  $([[ $CLEAN == '1' ]] && echo 'YES (overwrite all)' || echo 'NO (merge)')"
echo "Excludes:    .venv, __pycache__, .git, node_modules, etc."
echo "========================================"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY-RUN] Would copy source files from $SRC/ to $REMOTE_DEST:$REMOTE_PATH"
  echo "[DRY-RUN] Excluding: .venv, __pycache__, *.pyc, .git, node_modules"
  echo ""
  echo "[DRY-RUN] Files that would be transferred:"
  cd "$SRC" && find . -type f \
    ! -path './.venv/*' \
    ! -path './venv/*' \
    ! -path './__pycache__/*' \
    ! -path './.git/*' \
    ! -path './node_modules/*' \
    ! -name '*.pyc' \
    ! -name '.DS_Store' \
    | head -30
  exit 0
fi

# Clean remote directory if requested (overwrite mode)
if [[ "$CLEAN" == "1" ]]; then
  echo "Cleaning remote directory (overwrite mode)..."
  ssh $SSH_OPTS "$REMOTE_DEST" "rm -rf $REMOTE_PATH/* $REMOTE_PATH/.[!.]* 2>/dev/null || true"
fi

# Create remote directory if it doesn't exist
echo "Ensuring remote directory exists..."
ssh $SSH_OPTS "$REMOTE_DEST" "mkdir -p $REMOTE_PATH"

# Use tar over SSH to copy files with exclusions
# --no-mac-metadata prevents macOS from creating ._ AppleDouble files
echo "Copying source files (excluding .venv, __pycache__, .git, etc.)..."
cd "$SRC" && tar --no-mac-metadata $TAR_EXCLUDES -cf - . | ssh $SSH_OPTS "$REMOTE_DEST" "cd $REMOTE_PATH && tar -xf -"

echo "========================================"
echo "Done! Source files pushed to: $REMOTE_DEST:$REMOTE_PATH"
echo "========================================"

# Show what's on the remote
echo ""
echo "Remote contents of $REMOTE_PATH:"
ssh $SSH_OPTS "$REMOTE_DEST" "ls -la $REMOTE_PATH | head -20"
