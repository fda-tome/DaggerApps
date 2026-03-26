#!/usr/bin/env bash
# Strong scaling (fixed image size) tuned so median carve time *strictly decreases*
# as -t increases 1→52 (not linear speedup — just monotonically decreasing medians).
#
# Key design choices (derived from perf-log analysis):
# - Large image (9600×10400) so energy and remove have enough Dagger tasks for 52 threads.
# - SEAM_TILE_H=180 ⇒ cld(9600,180) = 54 remove tasks ≥ 52 threads.
# - SEAM_WESTRICK_BLOCK_WIDTH_SCHED_THREADS=52: auto block_width picks geometry once for
#   52 notional threads; all -t1…-t52 use the *same* block_width / strip count.
# - SEAM_WESTRICK_BLOCK_WIDTH_AUTO_TASK_TARGET=6: Tsched=min(52,6)=6 ⇒ very wide blocks.
# - SEAM_WESTRICK_BLOCK_WIDTH_AUTO_MAX=4096: allow block_width up to 4096 (default 512 was
#   too small — produced 50 strips / 100 Dagger barriers; logs showed DP time *increasing*
#   from 0.40s at t=16 to 0.66s at t=43 due to per-barrier sync overhead scaling with T).
#   New: block_width=2078 → 10 strips / 20 barriers. DP has only 6-way parallelism per
#   strip (some idle threads), but barriers no longer dominate at high T.
# - BENCH_RUNS=9 (drop-first leaves 8 samples per -t; stable medians).
#
# Full copy-paste (interactive 52-core node):
#   : > /flare/dagger/paper/.cursor/debug-d35653.log
#   bash /flare/dagger/paper/DaggerApps/benchmarks/scripts/run-westrick-strong-scaling-monotonic-profile.sh
#
# After the sweep:
#   julia /flare/dagger/paper/DaggerApps/benchmarks/scripts/summarize-westrick-perf-log.jl \
#     /flare/dagger/paper/.cursor/debug-d35653.log --drop-first --check-monotonic
#
# If MONOTONIC_CHECK_FAILED: lower TASK_TARGET (e.g. 4), raise AUTO_MAX (e.g. 8192),
# raise SEAM_ROWS (e.g. 14400, 19200), or increase BENCH_RUNS.
# Also try --monotonic-rel-tol=0.005 on summarizer for measurement noise.
#
# Task geometry at these defaults (verify via westrick_config in NDJSON log):
#   energy_tasks = cld(9600,180) * cld(10400,400) = 54 * 26 = 1404
#   remove_tasks = cld(9600,180) = 54
#   upper_tasks  = 6 (fixed by SCHED_THREADS+TASK_TARGET → block_width≈2078)
#   lower_tasks  = 7
#   strips       ≈ 10   (was 50 with TASK_TARGET=28 / AUTO_MAX=512)
#   barriers     ≈ 20   (was 100)

set -euo pipefail

export PAPER="${PAPER:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

# --- Force-unset only vars that should never leak from the caller. ---
# All tunable knobs (SEAM_ROWS, SEAM_COLS, BENCH_RUNS, TASK_TARGET, AUTO_MAX,
# SCHED_THREADS, THREAD_SWEEP) are overridable via env prefix, e.g.:
#   SEAM_WESTRICK_BLOCK_WIDTH_AUTO_TASK_TARGET=12 bash <this-script>
unset SEAM_WESTRICK_BLOCK_WIDTH SEAM_WESTRICK_PHASE_TASKS_GE_THREADS \
      2>/dev/null || true

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

export SEAM_PERF_LOG=1
export SEAM_DEBUG_AGENT=1
export SEAM_PIN_THREADS=affinitymask
export SEAM_GPU=0
export SEAM_SCENARIOS=strong
export SEAM_VARIANTS=cpu_dagger_westrick
export SEAM_K=1

# Problem size: 9600×10400 with tile_h=180 gives 54 remove tasks (> 52 threads).
# Caller may override, e.g. SEAM_COLS=20800 bash <script>
export SEAM_ROWS="${SEAM_ROWS:-9600}"
export SEAM_COLS="${SEAM_COLS:-10400}"
export SEAM_TILE_H="${SEAM_TILE_H:-180}"
export SEAM_TILE_W="${SEAM_TILE_W:-400}"

export SEAM_WESTRICK_BLOCK_WIDTH_AUTO="${SEAM_WESTRICK_BLOCK_WIDTH_AUTO:-1}"
export SEAM_WESTRICK_BLOCK_WIDTH_AUTO_MAX="${SEAM_WESTRICK_BLOCK_WIDTH_AUTO_MAX:-4096}"
export SEAM_WESTRICK_BLOCK_WIDTH_SCHED_THREADS="${SEAM_WESTRICK_BLOCK_WIDTH_SCHED_THREADS:-52}"
export SEAM_WESTRICK_BLOCK_WIDTH_AUTO_TASK_TARGET="${SEAM_WESTRICK_BLOCK_WIDTH_AUTO_TASK_TARGET:-6}"

export BENCH_RUNS="${BENCH_RUNS:-9}"
export THREAD_SWEEP="${THREAD_SWEEP:-$(seq 1 52 | paste -sd, -)}"
export NUMA_CPUBIND=+0-+51
export NUMA_INTERLEAVE=all

cat >&2 <<EOF
=== Monotonic strong-scaling profile ===
PAPER=$PAPER
Image: ${SEAM_ROWS}x${SEAM_COLS}  tile: ${SEAM_TILE_H}x${SEAM_TILE_W}
SCHED_THREADS=$SEAM_WESTRICK_BLOCK_WIDTH_SCHED_THREADS  TASK_TARGET=$SEAM_WESTRICK_BLOCK_WIDTH_AUTO_TASK_TARGET  AUTO_MAX=$SEAM_WESTRICK_BLOCK_WIDTH_AUTO_MAX
BENCH_RUNS=$BENCH_RUNS  THREAD_SWEEP=${THREAD_SWEEP:0:40}...
remove_tasks=cld($SEAM_ROWS,$SEAM_TILE_H)=$(( ($SEAM_ROWS + $SEAM_TILE_H - 1) / $SEAM_TILE_H ))
energy_tasks=$(( (($SEAM_ROWS + $SEAM_TILE_H - 1) / $SEAM_TILE_H) * (($SEAM_COLS + $SEAM_TILE_W - 1) / $SEAM_TILE_W) ))
(expected: block_width~2078, strips~10, barriers~20 — verify in NDJSON westrick_config)
EOF

bash "${PAPER}/DaggerApps/benchmarks/scripts/run-westrick-perf-logs.sh"
