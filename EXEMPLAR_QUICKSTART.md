# SC26 smoke path: checkable results (DaggerApps)

**Authoritative:** the **Artifact Description PDF** (paper repo `sc26_ad_submission.tex` → `sc26_ad_submission.pdf`) is canonical; this file is a clone-friendly copy. Update both when the frozen SHA changes.

## Golden path (reduced tier, commodity hardware)

1. **Clone and verify**
   ```bash
   git clone --branch SC26_AD_AE https://github.com/fda-tome/DaggerApps.git
   cd DaggerApps
   git checkout sc26-ad-freeze
   git rev-parse HEAD
   ```
   After `git checkout sc26-ad-freeze`, `git rev-parse HEAD` must match the
   **full commit SHA** in the SC26 Artifact Description PDF (tag `sc26-ad-freeze`
   on `SC26_AD_AE` should point to that commit on the remote after you push).

2. **Julia** — 1.12.4+ recommended (`juliaup add 1.12.4 && juliaup default 1.12.4`).

3. **Instantiate** every app the paper uses (from repo root):
   - `apps/gpu-cholesky`, `apps/barnes-hut`, `apps/seam-carving`, `apps/game-of-life`, `apps/heat-propagation`, `apps/pass-at-k-study`  
   ```bash
   for p in apps/gpu-cholesky apps/barnes-hut apps/seam-carving apps/game-of-life apps/heat-propagation apps/pass-at-k-study; do
     julia --project="$p" -e 'using Pkg; Pkg.instantiate()'
   done
   ```
   On HPC, load CUDA/ROCm/oneAPI modules **before** this step if you need GPU resolution.

4. **Reduced: all apps in one pass** (same code paths as the paper, smaller parameters — see `benchmarks/REVIEWER_PATHS.md`):
   ```bash
   bash benchmarks/run_smoke_all.sh
   ```
   The script must print: `=== run_smoke_all.sh finished ===`. Steps may log `[skip] ...` (e.g. Cholesky without a GPU — use `SKIP_CHOLESKY=1` to skip Cholesky on purpose on CPU-only hosts).

5. **Full paper tier (all apps)** — after you have the hardware/APIs for the original experiments:
   ```bash
   bash benchmarks/run_paper_all.sh
   ```
   Same drivers as (4); only env defaults change (bigger `SEAM_*`, 4× GPU for Cholesky, `run_full_study.sh` for pass@k, etc.). Details: `benchmarks/AD_BENCHMARKS.md` and `benchmarks/REVIEWER_PATHS.md`.

   **pass@k** is included in both scripts: reduced path calls `run_smoke_test.sh` (OpenAI → Ollama → static JSONL fallback so evaluate/analyze always run); full path calls `run_full_study.sh` and needs a real model + key or Ollama.

## What “success” means (reduced and full)

| Workload   | Check |
|------------|--------|
| Cholesky   | New results under `benchmarks/results/gpu-cholesky/` when a GPU stack is present; otherwise expect `[skip] gpu-cholesky` (or set `SKIP_CHOLESKY=1` on CPU-only hosts). |
| Barnes–Hut | `strong_scaling.csv` (or a run directory) under `benchmarks/results/barnes-hut/`. |
| Seam       | New outputs under `benchmarks/results/seam-carving/`. |
| Stencils   | New run dirs under `benchmarks/results/game-of-life/` and `.../heat-propagation/`. |
| Pass@k     | Stochastic: look for `outputs/`, `figures/`, or `tables/` under `apps/pass-at-k-study/` when the smoke path runs. |

LLM pass@*k* scores are **not** expected to match a fixed number across runs.

## Known issues

- **CUDA + `LD_LIBRARY_PATH`:** avoid prepending `/usr/local/cuda/lib64` when using CUDA.jl; `run_smoke_all.sh` / `run_paper_all.sh` strip it unless `CHOLESKY_KEEP_SYSTEM_CUDA_LD=1`.
- **Single-GPU Cholesky:** the driver may remap the default algorithm; see `AD_BENCHMARKS.md` and `CHOLESKY_FORCE_RL_LA`. **CPU-only:** export `SKIP_CHOLESKY=1` before `run_smoke_all.sh` so the rest of the apps still run in one session.
- **License:** see root `LICENSE` (MIT). Zenodo (optional at first AD) should archive the same tree as this tag.
