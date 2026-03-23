# Runner environment variables (Iris, Legate)

Julia’s `evaluate.jl` calls `src/utils/runners/runners.jl`. Each framework runs **shell commands** in a **temporary directory** where the generated source file is written.

## Placeholders in `*_BUILD_CMD` / `LEGATE_RUN_CMD`

- **`{SRC}`** — replaced by the task’s source filename (e.g. `l1_basic_spawn.cpp`).
- **`{SCRIPT}`** — same idea for Legate (usually `solution.py` or the task’s `.py` name).

If you set `IRIS_BUILD_CMD` **without** `{SRC}`, the runner **appends** ` <filename>` after your string (legacy style).

---

## This `paper` checkout: vendored Iris under `.runtime/opt`

If your clone includes **`$PAPER_ROOT/.runtime/opt/iris`** (headers + `lib64/*.so`), use the helper:

```bash
source /eagle/dagger/paper/DaggerApps/apps/pass-at-k-study/scripts/env_paper_runtime.sh
# or on Lustre canonical path, same tree:
# source /lus/eagle/projects/dagger/paper/DaggerApps/apps/pass-at-k-study/scripts/env_paper_runtime.sh
```

That sets (resolved from the script location):

| Tree | Typical absolute path (same on `/eagle/...` or `/lus/eagle/...` if shared) |
|------|-------------------------------------------------------------------------------|
| **Iris** | `$PAPER_ROOT/.runtime/opt/iris/include` (for `#include <iris/iris.h>`) |
| | `$PAPER_ROOT/.runtime/opt/iris/lib64/libiris.so` |

The script exports **`IRIS_BUILD_CMD`** with **`-I` / `-L` / `-Wl,-rpath,...`** pointing at those dirs, plus **`LD_LIBRARY_PATH`** for **`libiris.so`**.

---

## Legate (Python) — conda, not under `.runtime`

**Script:** `scripts/setup_legate_conda.sh` (see file header for `LEGATE_CONDA_*` variables).

| Variable | Purpose |
|----------|---------|
| **`LEGATE_RUN_CMD`** | Full run command, e.g. `"/path/bin/python solution.py"`, or use `{SCRIPT}` for the task filename. |
| **`LEGATE_PYTHON`** | Used when `LEGATE_RUN_CMD` is **unset**: runs `"$LEGATE_PYTHON" + " " + script`. |

**Example after `setup_legate_conda.sh`:**

```bash
export LEGATE_CONDA_PKGS_DIR=/eagle/dagger/paper/.conda_pkgs
bash scripts/setup_legate_conda.sh
export LEGATE_PYTHON="/eagle/dagger/paper/.conda/envs/legate/bin/python"
```

(Adjust `LEGATE_CONDA_PREFIX` / paths if your env lives elsewhere.)

---

## Iris without the helper (manual)

```bash
PAPER_ROOT=/eagle/dagger/paper   # or /lus/eagle/projects/dagger/paper

export IRIS_BUILD_CMD="g++ -std=c++17 -O2 -o main {SRC} -I\"${PAPER_ROOT}/.runtime/opt/iris/include\" -L\"${PAPER_ROOT}/.runtime/opt/iris/lib64\" -liris -Wl,-rpath,\"${PAPER_ROOT}/.runtime/opt/iris/lib64\""
export IRIS_RUN_CMD="./main"
```

If **`.runtime` is missing**, build/install Iris separately and substitute your site’s **include** / **lib** paths (modules, Spack, etc.).

---

## Quick checklist before `run_full_study.sh`

1. `source scripts/env_paper_runtime.sh` **or** set `IRIS_*` manually.
2. Legate: `setup_legate_conda.sh` → `export LEGATE_PYTHON=...`.
3. In the **same shell**: `julia ... src/evaluate.jl` (or full study).

---

## Suggested Ollama models (coding)

1. **`qwen3-coder:30b`** / **`30b-a3b-q8_0`**
2. **`qwen2.5-coder:32b`** — often better at **exact** function signatures (Dagger harness).
3. **`deepseek-coder:33b`**

```bash
ollama pull qwen2.5-coder:32b
```
