#!/usr/bin/env bash
# Paper-wide experiment launcher (wrapper around run_paper_experiments.py)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -x "$HOME/implementation/.venv_hosta/bin/python" ]]; then
  PYTHON="$HOME/implementation/.venv_hosta/bin/python"
elif [[ -x "$HOME/implementation/.venv/bin/python" ]]; then
  PYTHON="$HOME/implementation/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

exec "$PYTHON" "$ROOT/run_paper_experiments.py" "$@"
