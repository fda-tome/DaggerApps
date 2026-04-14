#!/usr/bin/env bash
set -euo pipefail

# Reproducible seam-carving scaling study (wavefront + triangles) with NUMA pinning.
# Defaults are chosen to match the implementation's nested-thread controls.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PROJECT="$ROOT_DIR/benchmarks/scripts"
RESULTS_DIR="$ROOT_DIR/benchmarks/results/seam-carving"
MANIFEST_DIR="$ROOT_DIR/benchmarks/results/seam-carving-study"
mkdir -p "$MANIFEST_DIR"

THREADS="${THREADS:-1 2 4 8 16 32 52}"
RUNS="${RUNS:-5}"
ROWS="${ROWS:-8192}"
COLS="${COLS:-8192}"
K="${K:-1}"
TILE_H="${TILE_H:-32}"
TILE_W="${TILE_W:-8192}"
SCENARIO="${SCENARIO:-strong}"    # strong | weak | both
WEAK_SCALE="${WEAK_SCALE:-sqrt}"  # only used when SCENARIO has weak
VARIANTS="${VARIANTS:-cpu_dagger_wavefront cpu_dagger_triangles}"

# Runtime controls inferred from seam implementation.
SEAM_DP_INNER_MIN_SPAN="${SEAM_DP_INNER_MIN_SPAN:-256}"
SEAM_DP_INNER_MIN_CHUNK="${SEAM_DP_INNER_MIN_CHUNK:-256}"
SEAM_DP_INNER_POOL_CAP="${SEAM_DP_INNER_POOL_CAP:-1}"
SEAM_CPU_NESTED_THREADS="${SEAM_CPU_NESTED_THREADS:-0}"
SEAM_GPU="${SEAM_GPU:-0}"
SEAM_DEVICE="${SEAM_DEVICE:-cpu}"

# NUMA pinning: cpuset-relative by default for scheduler-managed jobs.
NUMACTL_CMD="${NUMACTL_CMD:-numactl --cpunodebind=+0 --membind=+0}"

# Math helpers
ntx=$(( (COLS + TILE_W - 1) / TILE_W ))
WAVE_SIZE="${SEAM_WAVE_SIZE:-$ntx}"
outer=$(( WAVE_SIZE < ntx ? WAVE_SIZE : ntx ))
(( outer < 1 )) && outer=1

TS="$(date +%Y%m%d_%H%M%S)"
MANIFEST="$MANIFEST_DIR/${TS}.tsv"

echo -e "timestamp\tthread\tinner\twave_size\trows\tcols\ttile_h\ttile_w\tscenario\tvariants\tout_dir" > "$MANIFEST"

echo "== Seam Scaling Study =="
echo "project      : $SCRIPT_PROJECT"
echo "threads      : $THREADS"
echo "runs         : $RUNS"
echo "size         : ${ROWS}x${COLS}"
echo "tile         : ${TILE_H}x${TILE_W}"
echo "ntx          : $ntx"
echo "wave_size    : $WAVE_SIZE"
echo "outer        : $outer"
echo "scenario     : $SCENARIO"
echo "weak_scale   : $WEAK_SCALE"
echo "variants     : $VARIANTS"
echo "numactl      : $NUMACTL_CMD"
echo "manifest     : $MANIFEST"
echo

for t in $THREADS; do
  inner=$(( (t + outer - 1) / outer ))
  echo "=== t=$t inner=$inner ==="

  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export JULIA_EXCLUSIVE=1
  export SEAM_GPU
  export SEAM_DEVICE
  export SEAM_CPU_NESTED_THREADS
  export SEAM_DP_INNER_POOL_CAP
  export SEAM_DP_INNER_MIN_SPAN
  export SEAM_DP_INNER_MIN_CHUNK
  export SEAM_DP_INNER_THREADS="$inner"
  export SEAM_WAVE_SIZE="$WAVE_SIZE"
  export SEAM_WEAK_SCALE="$WEAK_SCALE"

  # Convert space-separated variant list to Julia symbols.
  julia_variants=()
  for v in $VARIANTS; do
    julia_variants+=(":$v")
  done
  variant_expr="[$(IFS=,; echo "${julia_variants[*]}")]"

  # Execute benchmark and capture output dir emitted by run_seam_carving.
  out="$($NUMACTL_CMD julia --project="$SCRIPT_PROJECT" -t"$t" -e '
    using DaggerAppsBenchmarks
    out = run_seam_carving(
      runs='