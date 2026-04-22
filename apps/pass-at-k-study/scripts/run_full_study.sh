#!/usr/bin/env bash
# Full pipeline for pass-at-k-study (same entrypoint for smoke vs paper tiers).
# Optional env (parameter-only tiering):
#   PASSK_SKIP_GENERATE=1     skip phase 1 (use existing JSONL)
#   PASSK_GENERATED=path      generated JSONL for phases 2–3 (required if skip)
#   PASSK_TASK=name           passed as --task to generate.jl
#   PASSK_OUTPUT=path         passed as --output to generate.jl
#   PASSK_API_BASE, PASSK_API_KEY  override API_BASE / key for generate (else API_BASE / OPENAI_API_KEY)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${1:-gpt-4o-mini}"
N_SAMPLES="${2:-5}"
API_BASE="${PASSK_API_BASE:-${API_BASE:-}}"
API_KEY="${PASSK_API_KEY:-${OPENAI_API_KEY:-}}"

GEN_EXTRA=()
if [ -n "${PASSK_TASK:-}" ]; then GEN_EXTRA+=(--task "$PASSK_TASK"); fi
if [ -n "${PASSK_OUTPUT:-}" ]; then GEN_EXTRA+=(--output "$PASSK_OUTPUT"); fi

echo "=== pass-at-k pipeline: model=$MODEL n_samples=$N_SAMPLES ==="

if [ "${PASSK_SKIP_GENERATE:-0}" != "1" ]; then
  echo "Phase 1: Generate..."
  julia --project=. src/generate.jl --model "$MODEL" --n-samples "$N_SAMPLES" \
    ${API_BASE:+--api-base "$API_BASE"} ${API_KEY:+--api-key "$API_KEY"} \
    "${GEN_EXTRA[@]}"
fi

GENERATED="${PASSK_GENERATED:-}"
if [ -z "$GENERATED" ]; then
  GENERATED=$(ls -t outputs/generated/*.jsonl 2>/dev/null | head -1)
fi
test -n "$GENERATED" || { echo "No generated file (set PASSK_GENERATED or run phase 1)."; exit 1; }
test -f "$GENERATED" || { echo "PASSK_GENERATED not found: $GENERATED"; exit 1; }

GEN_BASE="$(basename "$GENERATED")"
EVAL="outputs/evaluated/evaluated_${GEN_BASE}"

echo "Phase 2: Evaluate ($GENERATED)..."
julia --project=. src/evaluate.jl "$GENERATED"

test -f "$EVAL" || EVAL=$(ls -t outputs/evaluated/*.jsonl 2>/dev/null | head -1)
test -n "$EVAL" && test -f "$EVAL" || { echo "No evaluated file."; exit 1; }

echo "Phase 3: Analyze ($EVAL)..."
julia --project=. src/analyze.jl "$EVAL"

echo "=== Done. Check figures/*.png and tables/*.md ==="
