#!/bin/bash
# Run on Apple silicon, or observe the previous Terminal command without using its GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
repo="$PWD"
old_repo="$HOME/AgentRisk-llama"
mode="${1:-}"
if [[ -z "$mode" ]]; then
  echo "1) My original overnight command is already running (monitor only)"
  echo "2) I have not started training (start all five folds with early results)"
  read -r -p "Choose 1 or 2: " choice </dev/tty
  case "$choice" in
    1) mode=watch ;;
    2) mode=run ;;
    *) echo "Run again and choose 1 or 2." >&2; exit 2 ;;
  esac
fi
mkdir -p output
summary="$repo/output/meeting-summary.txt"
if [[ "$mode" == "watch" ]]; then
  exec python3 experiments/fivefold/meeting_report.py --watch \
    --root "$old_repo/output/llama-3b-fivefold" --output "$summary"
fi
if [[ "$mode" != "run" ]]; then
  echo "Use run for a new five-fold run or watch for the earlier command." >&2
  exit 2
fi
# The read-only reporter starts before package installation or downloading weights.
python3 experiments/fivefold/meeting_report.py --watch --output "$summary" > output/meeting-monitor.log 2>&1 &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
py="$repo/.venv-llama/bin/python"
if [[ ! -x "$py" && -x "$old_repo/.venv-llama/bin/python" ]]; then
  py="$old_repo/.venv-llama/bin/python"
fi
if [[ ! -x "$py" ]]; then
  for candidate in python3.12 python3.11 python3; do
    if "$candidate" -c 'import sys; assert sys.version_info[:2] in ((3,11), (3,12))' 2>/dev/null; then
      "$candidate" -m venv .venv-llama
      break
    fi
  done
  if [[ ! -x "$py" ]]; then
    python3 -m venv .venv-bootstrap
    .venv-bootstrap/bin/python -m pip install uv
    .venv-bootstrap/bin/uv venv --python 3.12 --seed .venv-llama
  fi
fi
"$py" -m pip install -r experiments/fivefold/requirements.txt
caffeinate -i "$py" -u experiments/fivefold/llama.py smoke 2>&1 | tee -a output/overnight.log
caffeinate -i "$py" -u experiments/fivefold/llama.py run --meeting 2>&1 | tee -a output/overnight.log
python3 experiments/fivefold/meeting_report.py --output "$summary"
