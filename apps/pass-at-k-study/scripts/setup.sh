#!/usr/bin/env bash
# One-time setup: install Julia dependencies for pass-at-k-study.
# Run from DaggerApps root: bash apps/pass-at-k-study/scripts/setup.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Julia dependencies ==="
julia --project=. -e 'using Pkg; Pkg.instantiate()'

echo "=== Setup done. Run: bash apps/pass-at-k-study/scripts/run_smoke_test.sh ==="
