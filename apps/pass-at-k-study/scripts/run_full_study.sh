#!/usr/bin/env bash
# Full pipeline for pass-at-k-study.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${1:-gpt-4o-mini}"
N_SAMPLES="${2:-5}"
API_BASE="${API_BASE:-}"
API_KEY="${OPENAI_API_KEY:-}"

echo "=== Full study: model=$MODEL n_samples=$N_SAMPLES ==="

echo "Phase 1: Generate..."
julia --project=. src/generate.jl --model "$MODEL" --n-samples "$N_SAMPLES" \
  ${API_BASE:+--api-base "$API_BASE"} ${API_KEY:+--api-key "$API_KEY"}
GENERATED=$(ls -t outputs/generated/*.jsonl 2>/dev/null | head -1)
test -n "$GENERATED" || { echo "No generated file."; exit 1; }

echo "Phase 2: Evaluate..."
julia --project=. src/evaluate.jl "$GENERATED"

echo "Phase 3: Analyze..."
EVAL=$(ls -t outputs/evaluated/*.jsonl 2>/dev/null | head -1)
test -n "$EVAL" || { echo "No evaluated file."; exit 1; }
julia --project=. src/analyze.jl "$EVAL"

echo "=== Done. Check figures/ and tables/ ==="
