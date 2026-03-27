#!/usr/bin/env bash
# Optional ROCm / device visibility for AMD GPU jobs (MI300A, MI250X, etc.).
# Source after your facility's `module load rocm` (or equivalent) if required.
#
# Usage:
#   source /path/to/DaggerApps/scripts/amd_rocm_env.sh
#
# Many sites expose 4–8 GPU devices per node; this benchmark expects **four**
# visible devices (see `DaggerGpuCholesky.four_gpu_processors`). Restrict with:
#   export HIP_VISIBLE_DEVICES=0,1,2,3
#   export ROCR_VISIBLE_DEVICES=0,1,2,3
#
# Julia AMDGPU typically picks up ROCm from the environment; no extra exports are
# strictly required beyond a working `hipcc` / `rocblas` stack.

if [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
  echo "amd_rocm_env: HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
fi
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
  echo "amd_rocm_env: ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES}"
fi
