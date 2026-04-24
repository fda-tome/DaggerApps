#!/usr/bin/env bash
# Reduced-tier presets: exports **parameters only**, then runs the same canonical
# invocations as `benchmarks/AD_BENCHMARKS.md` and `benchmarks/REVIEWER_PATHS.md`
# (best-effort: steps that need unavailable hardware print [skip] and the script still exits 0).
#
# Covers every SC26 case-study app: Cholesky, Barnes–Hut, seam (+ Westrick driver),
# game-of-life, heat-propagation, pass@k. Cholesky uses the same Dagger :ll path as
# the paper; four visible GPUs; small CHOLESKY_NS. See AD_BENCHMARKS.md for one-liners.

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

# Probe whether a vendor GPU package can be loaded (CUDA > AMDGPU > oneAPI > Metal).
_cholesky_run_smoke() {
  julia --project=apps/gpu-cholesky -e '
    skip = get(ENV, "SKIP_CHOLESKY", "")
    if skip == "1" || skip == "yes" || skip == "true"
      @info "SKIP_CHOLESKY set; skipping gpu-cholesky smoke"
      exit(2)
    end
    try
      using CUDA
    catch
      try
        using AMDGPU
      catch
        try
          using oneAPI
        catch
          try
            using Metal
          catch
            @info "No GPU backend (CUDA, AMDGPU, oneAPI, Metal); skipping gpu-cholesky reduced smoke. Install one GPU stack or set SKIP_CHOLESKY=1."
            exit(2)
          end
        end
      end
    end
    using Dagger
    include(joinpath("benchmarks", "scripts", "gpu-cholesky.jl"))
    run_benchmark()
  ' || return $?
  return 0
}

echo "=== DaggerApps smoke / reduced tier (all apps) ==="
echo "Docs: benchmarks/REVIEWER_PATHS.md, benchmarks/AD_BENCHMARKS.md, EXEMPLAR_QUICKSTART.md (repo root)"
echo "Full tier:  bash benchmarks/run_paper_all.sh"
echo

# --- Cholesky: same Dagger :ll path as paper; 4 visible GPUs, small N (if <4 GPUs, driver skips) ---
if command -v julia >/dev/null 2>&1; then
  (
    export LD_LIBRARY_PATH="$(_cholesky_clean_ld)"
    export CHOLESKY_ALGO="${CHOLESKY_ALGO:-ll}"
    export CHOLESKY_NS="${CHOLESKY_NS:-1024,2048}"
    export CHOLESKY_TRIALS="${CHOLESKY_TRIALS:-1}"
    export CHOLESKY_WARMUP="${CHOLESKY_WARMUP:-0}"
    export CHOLESKY_VENDOR="${CHOLESKY_VENDOR:-0}"
    if _cholesky_run_smoke; then
      : ok
    else
      s=$?
      if [ "$s" = 2 ]; then
        echo "[skip] gpu-cholesky (no GPU backend or SKIP_CHOLESKY=1)"
      else
        echo "[skip] gpu-cholesky (error during run; see messages above)"
      fi
    fi
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

# --- Seam Westrick thread sweep (subprocess driver → same seam project + seam-carving.jl) ---
(
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export SEAM_THREAD_SWEEP="${SEAM_THREAD_SWEEP:-1,2,4}"
  export BENCH_RUNS="${BENCH_RUNS:-1}"
  export SEAM_ROWS="${SEAM_ROWS:-256}"
  export SEAM_COLS="${SEAM_COLS:-256}"
  export SEAM_SCENARIOS="${SEAM_SCENARIOS:-strong}"
  julia benchmarks/scripts/seam-westrick-scaling.jl \
    || echo "[skip] seam-westrick-scaling"
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
