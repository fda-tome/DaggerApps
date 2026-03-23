#!/usr/bin/env bash
# Source this file to point Iris runners at the paper checkout's vendored
# `.runtime/opt/iris` tree (headers + shared lib).
#
# Usage (from anywhere):
#   source /path/to/DaggerApps/apps/pass-at-k-study/scripts/env_paper_runtime.sh
#
# Or from pass-at-k-study:
#   source scripts/env_paper_runtime.sh
#
# Then run evaluate / full study in the same shell.

set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# scripts -> pass-at-k-study -> apps -> DaggerApps -> paper (repo root containing .runtime)
PAPER_ROOT="$(cd "${_SCRIPT_DIR}/../../../.." && pwd)"
IRIS_ROOT="${PAPER_ROOT}/.runtime/opt/iris"

if [[ ! -d "${IRIS_ROOT}/include/iris" ]]; then
  echo "WARN: Iris tree missing at ${IRIS_ROOT} (expected ${IRIS_ROOT}/include/iris/iris.h)" >&2
fi

# {SRC} is replaced by the actual .cpp filename per task (see runners.jl).
export IRIS_BUILD_CMD="g++ -std=c++17 -O2 -o main {SRC} -I\"${IRIS_ROOT}/include\" -L\"${IRIS_ROOT}/lib64\" -liris -Wl,-rpath,\"${IRIS_ROOT}/lib64\""
export IRIS_RUN_CMD="./main"

# Help dynamic loader when subprocess cwd is a temp dir (rpath above usually enough).
export LD_LIBRARY_PATH="${IRIS_ROOT}/lib64:${LD_LIBRARY_PATH:-}"

echo "PAPER_ROOT=${PAPER_ROOT}"
echo "IRIS_ROOT=${IRIS_ROOT}"
echo "IRIS_BUILD_CMD set (with {SRC}); IRIS_RUN_CMD=./main"
echo "Legate: run scripts/setup_legate_conda.sh then export LEGATE_PYTHON or LEGATE_RUN_CMD (not vendored under .runtime)."
