#!/usr/bin/env bash
# Paper-tier presets: exports **parameters only**, then runs the same canonical
# invocations as `benchmarks/AD_BENCHMARKS.md`. Expects resources matching the AD (e.g. 4 GPUs for Cholesky).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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

echo "=== DaggerApps paper / full tier (parameter env only) ==="

# Clear pass@k smoke-only skip/generate overrides (keep PASSK_TASK / PASSK_OUTPUT if set for a scoped paper run)
unset PASSK_SKIP_GENERATE PASSK_GENERATED PASSK_API_BASE PASSK_API_KEY || true

# --- Cholesky ---
(
  export LD_LIBRARY_PATH="$(_cholesky_clean_ld)"
  export CHOLESKY_NUM_GPUS="${CHOLESKY_NUM_GPUS:-4}"
  export CHOLESKY_TRIALS="${CHOLESKY_TRIALS:-5}"
  export CHOLESKY_WARMUP="${CHOLESKY_WARMUP:-1}"
  export CHOLESKY_VENDOR="${CHOLESKY_VENDOR:-1}"
  julia --project=apps/gpu-cholesky -e 'using CUDA; using Dagger; include("benchmarks/scripts/gpu-cholesky.jl"); run_benchmark()' \
    || echo "[skip] gpu-cholesky (needs CUDA + GPUs matching CHOLESKY_NUM_GPUS)"
) || true

# --- Barnes (override BARNES_NPROCS to match your allocation) ---
(
  export BARNES_NPROCS="${BARNES_NPROCS:-16}"
  export BARNES_N_STRONG="${BARNES_N_STRONG:-250000}"
  export BENCH_RUNS="${BENCH_RUNS:-3}"
  export BENCH_WARMUP="${BENCH_WARMUP:-1}"
  julia --project=apps/barnes-hut -e 'using Dagger; include("benchmarks/scripts/barnes-hut.jl"); run_benchmark()'
) || true

# --- Seam ---
(
  export SEAM_ROWS="${SEAM_ROWS:-512}"
  export SEAM_COLS="${SEAM_COLS:-512}"
  export BENCH_RUNS="${BENCH_RUNS:-3}"
  export SEAM_SCENARIOS="${SEAM_SCENARIOS:-both}"
  julia --threads=auto --project=apps/seam-carving -e 'using Dagger; include("benchmarks/scripts/seam-carving.jl"); run_benchmark()' \
    || echo "[skip] seam-carving"
) || true

# --- Seam Westrick thread sweep ---
(
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export SEAM_THREAD_SWEEP="${SEAM_THREAD_SWEEP:-1,2,4,8,16,32,52}"
  export BENCH_RUNS="${BENCH_RUNS:-3}"
  export SEAM_ROWS="${SEAM_ROWS:-512}"
  export SEAM_COLS="${SEAM_COLS:-512}"
  export SEAM_SCENARIOS="${SEAM_SCENARIOS:-both}"
  julia benchmarks/scripts/seam-westrick-scaling.jl \
    || echo "[skip] seam-westrick-scaling"
) || true

# --- Game of life ---
(
  export LIFE_ROWS="${LIFE_ROWS:-1024}"
  export LIFE_COLS="${LIFE_COLS:-1024}"
  export LIFE_STEPS="${LIFE_STEPS:-100}"
  export BENCH_RUNS="${BENCH_RUNS:-3}"
  export LIFE_SCENARIOS="${LIFE_SCENARIOS:-both}"
  julia --project=apps/game-of-life -e 'using Dagger; include("benchmarks/scripts/game-of-life.jl"); run_benchmark()' \
    || echo "[skip] game-of-life"
) || true

# --- Heat ---
(
  export HEAT_ROWS="${HEAT_ROWS:-512}"
  export HEAT_COLS="${HEAT_COLS:-512}"
  export HEAT_STEPS="${HEAT_STEPS:-200}"
  export BENCH_RUNS="${BENCH_RUNS:-3}"
  export HEAT_SCENARIOS="${HEAT_SCENARIOS:-both}"
  julia --project=apps/heat-propagation -e 'using Dagger; include("benchmarks/scripts/heat-propagation.jl"); run_benchmark()' \
    || echo "[skip] heat-propagation"
) || true

# --- pass@k ---
(
  cd apps/pass-at-k-study
  if ! bash scripts/run_full_study.sh "${PASSK_MODEL:-gpt-4o-mini}" "${PASSK_N_SAMPLES:-5}"; then
    echo "[skip] pass-at-k-study (needs API keys / model)"
  fi
) || true

echo "=== run_paper_all.sh finished ==="
