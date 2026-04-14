#!/usr/bin/env bash
# Start Ollama server in the background with all 4 node GPUs visible (Polaris-style: 0–3).
# Safe to re-run: exits if the API already responds.
#
# Usage:
#   bash scripts/start_ollama_4gpu_background.sh
#   bash scripts/start_ollama_4gpu_background.sh deepseek-coder:33b   # also pull a model (foreground pull)
#   MODELS="a b c" bash scripts/start_ollama_4gpu_background.sh       # pull several after start
#
# Env:
#   CUDA_VISIBLE_DEVICES  default 0,1,2,3
#   OLLAMA_HOST           default 127.0.0.1:11434 (passed to ollama serve)
#   OLLAMA_LOG            default /tmp/ollama-${USER}.log
#   MODELS                space-separated list to `ollama pull` after server is up

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_LOG="${OLLAMA_LOG:-/tmp/ollama-${USER}.log}"

ollama_ping() {
  curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1
}

if ! command -v ollama &>/dev/null; then
  echo "ollama not in PATH."
  exit 1
fi

if ollama_ping; then
  echo "Ollama already up at http://${OLLAMA_HOST} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
else
  echo "Starting ollama serve in background — GPUs: ${CUDA_VISIBLE_DEVICES}"
  echo "Log: ${OLLAMA_LOG}"
  nohup ollama serve >>"${OLLAMA_LOG}" 2>&1 &
  echo "PID $!"
  for _ in $(seq 1 120); do
    ollama_ping && break
    sleep 1
  done
  ollama_ping || {
    echo "Ollama did not become ready. See: ${OLLAMA_LOG}"
    exit 1
  }
  echo "Ollama ready at http://${OLLAMA_HOST}"
fi

# Optional: pull model(s)
if [ -n "${MODELS:-}" ]; then
  for m in $MODELS; do
    echo "Pulling: $m"
    ollama pull "$m"
  done
elif [ "${1:-}" != "" ]; then
  echo "Pulling: $1"
  ollama pull "$1"
fi

echo "API base for generate.jl: http://${OLLAMA_HOST}/v1"
