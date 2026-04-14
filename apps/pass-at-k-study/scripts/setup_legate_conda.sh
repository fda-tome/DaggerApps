#!/usr/bin/env bash
# Install NVIDIA Legate via conda (official supported path). Requires conda >= 24.1.
# Docs: https://docs.nvidia.com/legate/latest/installation.html
#
# HPC / small $HOME: conda caches repodata and packages under CONDA_PKGS_DIRS (default:
# ~/miniforge3/pkgs). The *environment files* still install under envs_dirs (default:
# ~/miniforge3/envs/<name>) unless you use LEGATE_CONDA_PREFIX. Moving only the package cache
# to /eagle while keeping -n legate can still fail at "Verifying transaction" with errno 122
# because linking writes into $HOME. If LEGATE_CONDA_PKGS_DIR is set and LEGATE_CONDA_PREFIX
# is not, this script defaults the prefix to <parent-of-pkgs>/.conda/envs/<LEGATE_CONDA_ENV>.
#
# Optional env:
#   LEGATE_CONDA_ENV            conda env name (default: legate); used for named env or default prefix basename
#   LEGATE_CONDA_PREFIX         create env at this path (conda -p …) instead of -n …
#   LEGATE_CONDA_PKGS_DIR       exported as CONDA_PKGS_DIRS (large-quota directory)
#   LEGATE_CONDA_NO_AUTO_PREFIX set to 1 to skip auto LEGATE_CONDA_PREFIX when only pkgs dir is set
#   LEGATE_CONDA_TMPDIR         TMPDIR for conda (default: <project>/.conda/tmp when using pkgs or -p)
#
# On PBS/interactive GPU nodes, TMPDIR is often /var/tmp/pbs.* with a tiny quota. Conda can then
# fail at "Executing transaction" with errno 122 and roll back even if the env prefix is on /eagle.
#   LEGATE_PYTHON_VERSION  3.11 | 3.12 | 3.13 (default: 3.12)
#   CONDA_EXE              path to conda if not on PATH
#   CONDA_OVERRIDE_CUDA    e.g. 12.4 to force a GPU CUDA stack variant
#   MINIFORGE_PREFIX       where Miniforge lives if using default conda (default: $HOME/miniforge3)
#
# Example (ALCF Polaris–style: project space under /eagle, adjust to your allocation):
#   export LEGATE_CONDA_PKGS_DIR=/eagle/dagger/paper/.conda_pkgs
#   bash scripts/setup_legate_conda.sh
#   # (prefix defaults to /eagle/dagger/paper/.conda/envs/legate — set LEGATE_CONDA_PREFIX to override)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo ">>> $*"; }
warn() { echo "!!! $*" >&2; }

LEGATE_CONDA_ENV="${LEGATE_CONDA_ENV:-legate}"
LEGATE_PYTHON_VERSION="${LEGATE_PYTHON_VERSION:-3.12}"
MINIFORGE_PREFIX="${MINIFORGE_PREFIX:-${HOME}/miniforge3}"

resolve_conda() {
  if [ -n "${CONDA_EXE:-}" ] && [ -x "$CONDA_EXE" ]; then
    echo "$CONDA_EXE"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return
  fi
  if [ -x "${MINIFORGE_PREFIX}/bin/conda" ]; then
    echo "${MINIFORGE_PREFIX}/bin/conda"
    return
  fi
  echo ""
}

CONDA="$(resolve_conda)"
if [ -z "$CONDA" ]; then
  warn "conda not found. Install Miniforge (https://github.com/conda-forge/miniforge) or set CONDA_EXE."
  warn "Then re-run: bash ${SCRIPT_DIR}/setup_legate_conda.sh"
  exit 1
fi

if [ -n "${LEGATE_CONDA_PKGS_DIR:-}" ]; then
  mkdir -p "$LEGATE_CONDA_PKGS_DIR"
  export CONDA_PKGS_DIRS="$LEGATE_CONDA_PKGS_DIR"
  log "Using CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS} (avoids filling \$HOME quota)"
  if [ -z "${LEGATE_CONDA_PREFIX:-}" ] && [ -z "${LEGATE_CONDA_NO_AUTO_PREFIX:-}" ]; then
    PARENT="$(cd "$(dirname "$LEGATE_CONDA_PKGS_DIR")" && pwd)"
    LEGATE_CONDA_PREFIX="${PARENT}/.conda/envs/${LEGATE_CONDA_ENV}"
    log "LEGATE_CONDA_PREFIX unset → using ${LEGATE_CONDA_PREFIX}"
  fi
fi

log "Using conda: $CONDA"
"$CONDA" --version

CONDA_BASE="$("$CONDA" info --base)"

if [ -n "${LEGATE_CONDA_PREFIX:-}" ]; then
  ENV_DIR="${LEGATE_CONDA_PREFIX}"
  ENV_KIND=prefix
else
  ENV_DIR="${CONDA_BASE}/envs/${LEGATE_CONDA_ENV}"
  ENV_KIND=name
fi

# Avoid PBS /var/tmp (or other small TMPDIR) during conda execute/post-install.
if [ -z "${LEGATE_CONDA_TMPDIR:-}" ]; then
  if [ -n "${LEGATE_CONDA_PKGS_DIR:-}" ]; then
    LEGATE_CONDA_TMPDIR="$(cd "$(dirname "$LEGATE_CONDA_PKGS_DIR")" && pwd)/.conda/tmp"
  elif [ "$ENV_KIND" = prefix ]; then
    LEGATE_CONDA_TMPDIR="$(cd "$(dirname "$ENV_DIR")/.." && pwd)/tmp"
  fi
fi
if [ -n "${LEGATE_CONDA_TMPDIR:-}" ]; then
  mkdir -p "$LEGATE_CONDA_TMPDIR"
  export TMPDIR="$LEGATE_CONDA_TMPDIR"
  export TMP="$TMPDIR"
  export TEMP="$TMPDIR"
  # Keep conda/pip caches off $HOME when possible
  CONDA_CACHE_ROOT="$(cd "$(dirname "$LEGATE_CONDA_TMPDIR")" && pwd)"
  export XDG_CACHE_HOME="${CONDA_CACHE_ROOT}/xdg_cache"
  mkdir -p "$XDG_CACHE_HOME"
  log "Using TMPDIR=${TMPDIR} (PBS /var/tmp quota often breaks conda after verify)"
  log "Using XDG_CACHE_HOME=${XDG_CACHE_HOME}"
fi

if [ -n "${CONDA_OVERRIDE_CUDA:-}" ]; then
  export CONDA_OVERRIDE_CUDA
fi

legate_import_ok() {
  [ -x "${ENV_DIR}/bin/python" ] && "${ENV_DIR}/bin/python" -c "import legate" >/dev/null 2>&1
}

conda_install_legate() {
  if [ "$ENV_KIND" = prefix ]; then
    mkdir -p "$(dirname "$ENV_DIR")"
    "$CONDA" install -y -p "$ENV_DIR" -c conda-forge -c legate \
      "python=${LEGATE_PYTHON_VERSION}" legate
  else
    "$CONDA" install -y -n "$LEGATE_CONDA_ENV" -c conda-forge -c legate \
      "python=${LEGATE_PYTHON_VERSION}" legate
  fi
}

conda_create_legate() {
  if [ "$ENV_KIND" = prefix ]; then
    mkdir -p "$(dirname "$ENV_DIR")"
    "$CONDA" create -y -p "$ENV_DIR" -c conda-forge -c legate \
      "python=${LEGATE_PYTHON_VERSION}" legate
  else
    "$CONDA" create -y -n "$LEGATE_CONDA_ENV" -c conda-forge -c legate \
      "python=${LEGATE_PYTHON_VERSION}" legate
  fi
}

if legate_import_ok; then
  log "Legate already importable at ${ENV_DIR}; nothing to do."
elif [ -d "$ENV_DIR" ]; then
  log "Env exists at ${ENV_DIR}; installing/upgrading legate..."
  conda_install_legate
else
  log "Creating legate env at ${ENV_DIR} (large download)..."
  conda_create_legate
fi

LEGATE_PY="${ENV_DIR}/bin/python"

if ! "$LEGATE_PY" -c "import legate" >/dev/null 2>&1; then
  warn "legate import still failing; check conda output above."
  warn "If you saw 'Disk quota exceeded' (errno 122): use LEGATE_CONDA_PKGS_DIR + prefix on /eagle,"
  warn "set LEGATE_CONDA_TMPDIR to a directory on that filesystem (script sets a default),"
  warn "and rm -rf the partial env directory before retrying."
  exit 1
fi

log "Legate OK."
cat <<EOF

Use this Python when running pass-at-k-study evaluations (Julia inherits env):

  export LEGATE_PYTHON="${LEGATE_PY}"

Or override the full run command per task directory:

  export LEGATE_RUN_CMD="${LEGATE_PY} solution.py"

EOF
