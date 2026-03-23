#!/usr/bin/env bash
# Full pass@k study with Ollama: start server in background if needed, then run pipeline.
#
# Usage (from DaggerApps repo root):
#   bash apps/pass-at-k-study/scripts/run_full_study_ollama.sh
#   bash apps/pass-at-k-study/scripts/run_full_study_ollama.sh "deepseek-coder:33b" 5
#
# Env:
#   OLLAMA_HOST          — bind address (default 127.0.0.1:11434); passed to `ollama serve`
#   OLLAMA_LOG           — server log file (default /tmp/ollama-${USER}.log)
#   CUDA_VISIBLE_DEVICES — GPUs visible to Ollama and to Julia (evaluate). Default 0,1,2,3.
#                          Set before running if you need a different set; restart Ollama if you change it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Same idea as scripts/setup_ollama.sh: make all node GPUs visible unless the user already set this.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

MODEL="${1:-deepseek-coder:33b}"
N_SAMPLES="${2:-5}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_LOG="${OLLAMA_LOG:-/tmp/ollama-${USER}.log}"

ollama_ping() {
  curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1
}

if ! command -v ollama &>/dev/null; then
  echo "ollama not in PATH. Install or run: bash scripts/setup_ollama.sh"
  exit 1
fi

if ! ollama_ping; then
  echo "Starting Ollama in background (log: $OLLAMA_LOG)..."
  export OLLAMA_HOST
  nohup ollama serve >>"$OLLAMA_LOG" 2>&1 &
  echo "Ollama PID $! — tail -f '$OLLAMA_LOG' to watch the server"
fi

echo "Waiting for Ollama at http://${OLLAMA_HOST} ..."
for _ in $(seq 1 60); do
  ollama_ping && break
  sleep 1
done
ollama_ping || {
  echo "Ollama did not become ready. See: $OLLAMA_LOG"
  exit 1
}

echo "Ensuring model is available: $MODEL"
ollama pull "$MODEL" || true

export API_BASE="http://${OLLAMA_HOST}/v1"
bash "$SCRIPT_DIR/run_full_study.sh" "$MODEL" "$N_SAMPLES"

echo "Ollama is still running in the background. Stop with: pkill -f 'ollama serve' (use care on shared machines)"
