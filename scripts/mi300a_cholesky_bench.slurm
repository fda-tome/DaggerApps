#!/bin/bash
# SLURM harness for Dagger GPU Cholesky on AMD MI300-class nodes (ROCm + AMDGPU.jl).
# Adjust account, partition, GPU count, and paths for your facility.
#
# Examples:
#   sbatch scripts/mi300a_cholesky_bench.slurm
#
# Typical prep (site-specific):
#   module load rocm
#   export JULIA_DEPOT_PATH="$SCRATCH/.julia"
#   source scripts/amd_rocm_env.sh   # optional device pinning
#
#SBATCH -J dagger-chol-amd
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -o dagger_chol_amd_%j.out
#SBATCH -e dagger_chol_amd_%j.err
##SBATCH -A YOUR_ALLOCATION
##SBATCH -p batch

set -euo pipefail

: "${PROJ:?Set PROJ to the parent directory that contains DaggerApps (e.g. export PROJ=$PWD)}"
cd "${PROJ}"

JULIA="${JULIA:-julia}"
JULIA_PROJECT="${JULIA_PROJECT:-${PROJ}/DaggerApps/apps/gpu-cholesky}"
LOGDIR="${LOGDIR:-${PROJ}/DaggerApps/benchmarks/results/gpu-cholesky}"
mkdir -p "${LOGDIR}"

export TMPDIR="${TMPDIR:-${HOME}/.julia/tmp}"
mkdir -p "${TMPDIR}"

export CHOLESKY_DEVICE="${CHOLESKY_DEVICE:-amdgpu}"

echo "=== Dagger GPU Cholesky (AMD / ROCm) ==="
echo "Host:    $(hostname)"
echo "Date:    $(date -Iseconds)"
echo "Julia:   $($JULIA --version)"
echo "CHOLESKY_DEVICE=${CHOLESKY_DEVICE}"
rocm-smi --showproductname 2>/dev/null || true
echo "========================================"

run_one() {
  local LABEL="$1" NS="$2" BS="$3" ALGO="$4" TRIALS="$5"
  shift 5
  local LOGFILE="${LOGDIR}/chol_amd_${LABEL}.log"

  if [ -f "$LOGFILE" ] && grep -q "^DONE:${LABEL}$" "$LOGFILE"; then
    echo ""
    echo "===== $LABEL  SKIP (already completed) ====="
    return 0
  fi

  echo ""
  echo "===== $LABEL  (N=$NS bs=$BS algo=$ALGO trials=$TRIALS) ====="

  (
    export CHOLESKY_NS="$NS"
    export CHOLESKY_BLOCK="$BS"
    export CHOLESKY_ALGO="$ALGO"
    export CHOLESKY_TRIALS="$TRIALS"
    for kv in "$@"; do export "$kv"; done

    "$JULIA" --project="$JULIA_PROJECT" -e '
      using AMDGPU
      using Dagger
      include("DaggerApps/benchmarks/scripts/gpu-cholesky.jl")
      run_benchmark()
    '
  ) > "$LOGFILE" 2>&1

  if [ $? -eq 0 ]; then
    echo "DONE:${LABEL}" >> "$LOGFILE"
    echo "  OK: $LABEL"
  else
    echo "  *** FAILED: $LABEL ***"
  fi
}

# ============ Same sweep as scripts/polaris_cholesky_bench.pbs (AMDGPU instead of CUDA) ============

# ============ 2×2 tile grid (block_size = N/2) ============
run_one N4096_2x2_rl_la   4096   2048  rl_la  5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N4096_2x2_ll      4096   2048  ll     5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N8192_2x2_rl_la   8192   4096  rl_la  5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N8192_2x2_ll      8192   4096  ll     5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N16384_2x2_rl_la  16384  8192  rl_la  5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N16384_2x2_ll     16384  8192  ll     5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N32768_2x2_rl_la  32768  16384 rl_la  5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N32768_2x2_ll     32768  16384 ll     5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N65536_2x2_rl_la  65536  32768 rl_la  5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N65536_2x2_ll     65536  32768 ll     5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N131072_2x2_rl_la 131072 65536 rl_la  3  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=1 CHOLESKY_PERF_LOG=1
run_one N131072_2x2_ll    131072 65536 ll     3  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=1 CHOLESKY_PERF_LOG=1

# ============ 4×4 tile grid (block_size = N/4) ============
run_one N4096_4x4_rl_la   4096   1024  rl_la  5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N4096_4x4_ll      4096   1024  ll     5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N8192_4x4_rl_la   8192   2048  rl_la  5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N8192_4x4_ll      8192   2048  ll     5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N16384_4x4_rl_la  16384  4096  rl_la  5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N16384_4x4_ll     16384  4096  ll     5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N32768_4x4_rl_la  32768  8192  rl_la  5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N32768_4x4_ll     32768  8192  ll     5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N65536_4x4_rl_la  65536  16384 rl_la  5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N65536_4x4_ll     65536  16384 ll     5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N131072_4x4_rl_la 131072 32768 rl_la  3  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=1 CHOLESKY_PERF_LOG=1
run_one N131072_4x4_ll    131072 32768 ll     3  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=1 CHOLESKY_PERF_LOG=1

# ============ 8×8 tile grid (block_size = N/8) ============
run_one N4096_8x8_rl_la   4096   512   rl_la  5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N4096_8x8_ll      4096   512   ll     5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N8192_8x8_rl_la   8192   1024  rl_la  5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N8192_8x8_ll      8192   1024  ll     5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N16384_8x8_rl_la  16384  2048  rl_la  5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N16384_8x8_ll     16384  2048  ll     5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N32768_8x8_rl_la  32768  4096  rl_la  5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N32768_8x8_ll     32768  4096  ll     5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N65536_8x8_rl_la  65536  8192  rl_la  5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N65536_8x8_ll     65536  8192  ll     5  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=2 CHOLESKY_PERF_LOG=1
run_one N131072_8x8_rl_la 131072 16384 rl_la  3  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=1 CHOLESKY_PERF_LOG=1
run_one N131072_8x8_ll    131072 16384 ll     3  CHOLESKY_VENDOR=1 CHOLESKY_WARMUP=1 CHOLESKY_PERF_LOG=1

echo ""
echo "=== All runs complete: $(date -Iseconds) ==="
echo "Logs: ${LOGDIR}/chol_amd_*.log"
