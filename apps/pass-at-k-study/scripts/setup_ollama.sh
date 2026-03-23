#!/usr/bin/env bash
# Install Ollama and pull models for local generation.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Prefer user-space Ollama first.
INSTALL_DIR="${HOME}/.local/bin"
export PATH="${INSTALL_DIR}:$PATH"
export LD_LIBRARY_PATH="${HOME}/.local/lib/ollama:${LD_LIBRARY_PATH:-}"

need_install=0
if ! command -v ollama &>/dev/null; then
  need_install=1
elif ! ollama --version >/dev/null 2>&1; then
  echo "Found ollama in PATH, but it is not runnable. Reinstalling user-space binary..."
  need_install=1
fi

if [ "$need_install" -eq 1 ]; then
  echo "Installing Ollama (user-space, no sudo)..."
  OLLAMA_BIN="${INSTALL_DIR}/ollama"
  mkdir -p "$INSTALL_DIR"

  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64|amd64)
      OLLAMA_BASENAME="ollama-linux-amd64"
      ;;
    aarch64|arm64)
      OLLAMA_BASENAME="ollama-linux-arm64"
      ;;
    *)
      echo "Unsupported architecture: ${ARCH}"
      echo "Please install Ollama manually for this architecture."
      exit 1
      ;;
  esac

  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' EXIT
  if curl -fsI "https://ollama.com/download/${OLLAMA_BASENAME}.tar.zst" >/dev/null 2>&1; then
    if ! command -v zstd >/dev/null 2>&1; then
      echo "Ollama package is .tar.zst but zstd is not installed."
      echo "Install zstd or use the official installer on a machine with sudo."
      exit 1
    fi
    ZST_PATH="${TMP_DIR}/ollama.tar.zst"
    curl -fL "https://ollama.com/download/${OLLAMA_BASENAME}.tar.zst" -o "$ZST_PATH"
    zstd -d "$ZST_PATH" -o "${TMP_DIR}/ollama.tar"
    tar -xf "${TMP_DIR}/ollama.tar" -C "$TMP_DIR"
  elif curl -fsI "https://ollama.com/download/${OLLAMA_BASENAME}.tgz" >/dev/null 2>&1; then
    TGZ_PATH="${TMP_DIR}/ollama.tgz"
    curl -fL "https://ollama.com/download/${OLLAMA_BASENAME}.tgz" -o "$TGZ_PATH"
    tar -xzf "$TGZ_PATH" -C "$TMP_DIR"
  else
    echo "Could not find downloadable Ollama archive for ${OLLAMA_BASENAME}."
    echo "Tried .tar.zst and .tgz from https://ollama.com/download/"
    exit 1
  fi

  # Tarball contains bin/ollama and lib/ollama; install both in user space.
  install -m 755 "${TMP_DIR}/bin/ollama" "$OLLAMA_BIN"
  mkdir -p "${HOME}/.local/lib"
  rm -rf "${HOME}/.local/lib/ollama"
  cp -r "${TMP_DIR}/lib/ollama" "${HOME}/.local/lib/ollama"

  if ! command -v ollama &>/dev/null; then
    echo "Ollama installed to ${OLLAMA_BIN}, but not in current PATH."
    echo "Run: export PATH=\"${INSTALL_DIR}:\$PATH\""
    exit 1
  elif ! ollama --version >/dev/null 2>&1; then
    echo "Downloaded ollama binary is not runnable."
    echo "Check proxy/network and verify archive availability for ${OLLAMA_BASENAME}."
    exit 1
  fi
fi

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama server in background..."
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  nohup ollama serve > "/tmp/ollama-${USER}.log" 2>&1 &
  sleep 3
fi

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Ollama server is not reachable at http://127.0.0.1:11434"
  echo "Check log: /tmp/ollama-${USER}.log"
  exit 1
fi

echo "Pulling models (e.g. codellama, deepseek-coder)..."
ollama pull codellama || true
ollama pull deepseek-coder || true
ollama pull qwen2.5-coder || true

echo "Ollama server is running at http://127.0.0.1:11434"
echo "Then run generation with: julia --project=. src/generate.jl --model codellama --api-base http://localhost:11434/v1"
