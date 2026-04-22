#!/usr/bin/env bash
# Reduced-tier presets: exports **parameters only**, then runs the same canonical
# invocations as `benchmarks/AD_BENCHMARKS.md` (best-effort; skip steps that need unavailable hardware).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Avoid mixing /usr/local/cuda into the dynamic loader with CUDA.jl's own CUDA
# stack (causes illegal memory access / heap corruption on some hosts).
_cholesky_clean_ld() {
  if [ "${CHOLESKY_KEEP_SYSTEM_CUDA_LD:-0}" = "1" ]; then
    echo "${LD_LIBRARY_PATH:-}"
    return
  fi
  if [ -z "${LD_LIBRARY_PATH:-}" ]; then
    echo ""
    return
  fi
  echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -vE '^/usr/local/cuda(/|$)' | paste -sd: -
}

echo "=== DaggerApps smoke / reduced tier (parameter env only) ==="

# --- Cholesky (Dagger path; 1 GPU if visible) ---
if command -v julia >/dev/null 2>&1; then
  (
    export LD_LIBRARY_PATH="$(_cholesky_clean_ld)"
    export CHOLESKY_NUM_GPUS="${CHOLESKY_NUM_GPUS:-1}"
    export CHOLESKY_NS="${CHOLESKY_NS:-1024,2048}"
    export CHOLESKY_TRIALS="${CHOLESKY_TRIALS:-1}"
    export CHOLESKY_WARMUP="${CHOLESKY_WARMUP:-0}"
    export CHOLESKY_VENDOR="${CHOLESKY_VENDOR:-0}"
    julia --project=apps/gpu-cholesky -e 'using CUDA; using Dagger; include("benchmarks/scripts/gpu-cholesky.jl"); run_benchmark()' \
      || echo "[skip] gpu-cholesky (CUDA not available or run failed)"
  ) || true
else
  echo "[skip] julia not in PATH"
fi

# --- Barnes (single process: set BARNES_NPROCS) ---
(
  export BARNES_NPROCS="${BARNES_NPROCS:-4}"
  export BARNES_N_STRONG="${BARNES_N_STRONG:-50000}"
  export BENCH_RUNS="${BENCH_RUNS:-1}"
  export BENCH_WARMUP="${BENCH_WARMUP:-0}"
  export BENCH_BT_SAMPLES="${BENCH_BT_SAMPLES:-1}"
  julia --project=apps/barnes-hut -e 'using Dagger; include("benchmarks/scripts/barnes-hut.jl"); run_benchmark()' \
    || echo "[skip] barnes-hut"
) || true

# --- Seam ---
(
  export SEAM_ROWS="${SEAM_ROWS:-256}"
  export SEAM_COLS="${SEAM_COLS:-256}"
  export BENCH_RUNS="${BENCH_RUNS:-1}"
  export SEAM_K="${SEAM_K:-1}"
  export SEAM_SCENARIOS="${SEAM_SCENARIOS:-strong}"
  julia --threads=auto --project=apps/seam-carving -e 'using Dagger; include("benchmarks/scripts/seam-carving.jl"); run_benchmark()' \
    || echo "[skip] seam-carving"
) || true

# --- Game of life ---
(
  export LIFE_ROWS="${LIFE_ROWS:-512}"
  export LIFE_COLS="${LIFE_COLS:-512}"
  export LIFE_STEPS="${LIFE_STEPS:-20}"
  export BENCH_RUNS="${BENCH_RUNS:-1}"
  export LIFE_SCENARIOS="${LIFE_SCENARIOS:-strong}"
  julia --project=apps/game-of-life -e 'using Dagger; include("benchmarks/scripts/game-of-life.jl"); run_benchmark()' \
    || echo "[skip] game-of-life"
) || true

# --- Heat ---
(
  export HEAT_ROWS="${HEAT_ROWS:-512}"
  export HEAT_COLS="${HEAT_COLS:-512}"
  export HEAT_STEPS="${HEAT_STEPS:-50}"
  export BENCH_RUNS="${BENCH_RUNS:-1}"
  export HEAT_SCENARIOS="${HEAT_SCENARIOS:-strong}"
  julia --project=apps/heat-propagation -e 'using Dagger; include("benchmarks/scripts/heat-propagation.jl"); run_benchmark()' \
    || echo "[skip] heat-propagation"
) || true

# --- pass@k (thin wrapper → run_full_study.sh) ---
if [ -f "$ROOT/apps/pass-at-k-study/scripts/run_smoke_test.sh" ]; then
  bash "$ROOT/apps/pass-at-k-study/scripts/run_smoke_test.sh" \
    || echo "[skip] pass-at-k-study smoke"
fi

echo "=== run_smoke_all.sh finished ==="
