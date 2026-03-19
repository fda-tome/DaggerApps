#!/usr/bin/env bash
# Install Ollama and pull models for local generation.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v ollama &>/dev/null; then
  echo "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "Pulling models (e.g. codellama, deepseek-coder)..."
ollama pull codellama || true
ollama pull deepseek-coder || true
ollama pull qwen2.5-coder || true

echo "Start Ollama with: ollama serve"
echo "Then run generation with: julia --project=. src/generate.jl --model codellama --api-base http://localhost:11434/v1"
