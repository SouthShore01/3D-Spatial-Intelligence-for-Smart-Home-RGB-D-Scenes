#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
  echo "No .venv found. Create one first:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -r requirements.txt"
  exit 1
fi

if [ ! -d "SUNRGBD" ]; then
  echo "SUNRGBD folder not found next to the bundle."
  echo "Expected: $ROOT_DIR/SUNRGBD"
  exit 1
fi

exec .venv/bin/python -m smart_home_rgbd.cli label-ui --port 8765
