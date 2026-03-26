#!/usr/bin/env bash
# Full strong-scaling + NDJSON perf logs for cpu_dagger_westrick.
# Logs append to: <paper>/DaggerApps/../.cursor/debug-d35653.log (repo .cursor)
#
# Usage (compute node, from anywhere):
#   bash /path/to/DaggerApps/benchmarks/scripts/run-westrick-perf-logs.sh
# Optional env:
#   THREAD_SWEEP="1,2,3,...,52"                 # default: all integers 1..52 (one subprocess per -t)
#   NUMA_CPUBIND="+0-+51"                         # empty = no numactl
#   NUMA_MEMBIND="0"
#   SEAM_ROWS SEAM_COLS SEAM_TILE_H SEAM_TILE_W BENCH_RUNS  # override sizes
#   SEAM_WESTRICK_BLOCK_WIDTH_AUTO=0|1
#   SEAM_WESTRICK_BLOCK_WIDTH_SCHED_THREADS=N  # pin Westrick strip geometry across thread sweep

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SEAM_PROJECT="${PAPER}/DaggerApps/apps/seam-carving"
BENCH="${PAPER}/DaggerApps/benchmarks/scripts/seam-carving.jl"
LOG="${PAPER}/.cursor/debug-d35653.log"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

export SEAM_PERF_LOG="${SEAM_PERF_LOG:-1}"
export SEAM_DEBUG_AGENT="${SEAM_DEBUG_AGENT:-1}"

export BENCH_RUNS="${BENCH_RUNS:-3}"
export SEAM_GPU="${SEAM_GPU:-0}"
export SEAM_SCENARIOS="${SEAM_SCENARIOS:-strong}"
export SEAM_VARIANTS="${SEAM_VARIANTS:-cpu_dagger_westrick}"
export SEAM_ROWS="${SEAM_ROWS:-2400}"
export SEAM_COLS="${SEAM_COLS:-10400}"
export SEAM_K="${SEAM_K:-1}"
export SEAM_TILE_H="${SEAM_TILE_H:-400}"
export SEAM_TILE_W="${SEAM_TILE_W:-400}"
export SEAM_WESTRICK_BLOCK_WIDTH_AUTO="${SEAM_WESTRICK_BLOCK_WIDTH_AUTO:-1}"

mkdir -p "$(dirname "$LOG")"
: >>"$LOG"
echo "Perf NDJSON log: $LOG" >&2

# Full strong-scaling sweep 1..52 when THREAD_SWEEP is unset or empty (comma-separated list allowed).
THREAD_SWEEP="${THREAD_SWEEP:-$(seq 1 52 | paste -sd, -)}"
IFS=',' read -r -a SWEEP <<< "${THREAD_SWEEP}"

NUMA_CPUBIND="${NUMA_CPUBIND:-+0-+51}"
NUMA_MEMBIND="${NUMA_MEMBIND:-0}"
# Set NUMA_INTERLEAVE=all to use --interleave=all instead of --membind.
# Recommended for 2-socket nodes where T > cores-per-socket.
NUMA_INTERLEAVE="${NUMA_INTERLEAVE:-}"

for t in "${SWEEP[@]}"; do
  t="$(echo "$t" | tr -d '[:space:]')"
  [[ -z "$t" ]] && continue
  echo "=== perf logs: -t$t (SEAM_PERF_LOG=$SEAM_PERF_LOG) ===" >&2
  if [[ -n "$NUMA_CPUBIND" ]]; then
    if [[ -n "$NUMA_INTERLEAVE" ]]; then
      numactl --physcpubind="$NUMA_CPUBIND" --interleave="$NUMA_INTERLEAVE" \
        julia --project="$SEAM_PROJECT" -t"$t" "$BENCH"
    else
      numactl --physcpubind="$NUMA_CPUBIND" --membind="$NUMA_MEMBIND" \
        julia --project="$SEAM_PROJECT" -t"$t" "$BENCH"
    fi
  else
    julia --project="$SEAM_PROJECT" -t"$t" "$BENCH"
  fi
done

echo "Done. NDJSON log: $LOG" >&2
echo "Scaling table (optional --drop-first to skip first carve per subprocess):" >&2
echo "  julia ${SCRIPT_DIR}/summarize-westrick-perf-log.jl \"$LOG\" --drop-first" >&2
echo "  julia ${SCRIPT_DIR}/summarize-westrick-perf-log.jl \"$LOG\" --drop-first --csv" >&2
echo "  julia ${SCRIPT_DIR}/summarize-westrick-perf-log.jl \"$LOG\" --drop-first --check-monotonic" >&2
