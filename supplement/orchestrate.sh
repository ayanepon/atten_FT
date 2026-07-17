#!/usr/bin/env bash
# Thin wrapper around orchestrate.py
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -x "$HOME/implementation/.venv_hosta/bin/python" ]]; then
  # hosta
  PYTHON="$HOME/implementation/.venv_hosta/bin/python"
elif [[ -x "$HOME/implementation/.venv/bin/python" ]]; then
  # hostb / generic
  PYTHON="$HOME/implementation/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

exec "$PYTHON" "$ROOT/orchestrate.py" "$@"
