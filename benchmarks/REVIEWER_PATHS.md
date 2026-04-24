# Reviewer paths — reduced (AE) vs full (paper)

This repository ships **one canonical driver per case study** (`benchmarks/scripts/*.jl` and `pass-at-k` shell entrypoints). The **only** difference between an **artifact-evaluation (reduced)** run and a **paper-scale (full)** run is **environment variables and resource allocation**—the same code paths execute.

| Goal | One command (from **repo root** after `Pkg.instantiate` on each `apps/*` project) |
|------|----------------------------------------------------------------------------------|
| **Run every app, reduced** | `bash benchmarks/run_smoke_all.sh` |
| **Run every app, full** | `bash benchmarks/run_paper_all.sh` |

Detailed one-liners and env tables: **`AD_BENCHMARKS.md`**. High-level setup: **`EXEMPLAR_QUICKSTART.md`** (repo root).

---

## What “reduced” and “full” mean

| Tier | Intent |
|------|--------|
| **Reduced** | Small problem sizes, few trials, few threads/processes, **optional** GPU (Cholesky uses 1 GPU and small `N` if a GPU stack is present). Confirms: Julia projects resolve, Dagger loads, drivers write results under `benchmarks/results/…` and (for pass@k) under `apps/pass-at-k-study/outputs/…`. |
| **Full** | Parameters aligned with the SC26 paper / Artifact Description: multi-GPU Cholesky where applicable, large seam image, longer stencil runs, `run_full_study.sh` for pass@k, etc. |

A clean **reduced** run is strong evidence the **same drivers** will work at **full** scale **on hardware that matches the paper** (memory, GPU count, cluster, API keys). It is not a mathematical guarantee—OOM, time limits, or missing API keys can still fail a full run.

---

## Per-app quick reference

Run from **repository root** (directory containing `apps/` and `benchmarks/`). Always `using Pkg; Pkg.instantiate()` for that `apps/<name>` first.

| App | Reduced (copy-paste) | Full (copy-paste) | Notes |
|-----|------------------------|-------------------|--------|
| **GPU Cholesky** | `SKIP_CHOLESKY=1` skips if you have no GPU. Otherwise same as smoke script block in `run_smoke_all.sh` (1 GPU, small `CHOLESKY_NS`). | Same pattern as `run_paper_all.sh` (default 4 GPUs, long trials). | Loads **CUDA**, else **AMDGPU**, else **oneAPI**, else **Metal** before `Dagger`. |
| **Barnes–Hut** | `BARNES_NPROCS=4 BARNES_N_STRONG=50000` + `barnes-hut.jl` | `BARNES_NPROCS=16 BARNES_N_STRONG=250000` (override for your cluster) | See `AD_BENCHMARKS.md` for exact `julia -e` lines. |
| **Seam carving** | `SEAM_ROWS=256 SEAM_COLS=256` + `seam-carving.jl` | `SEAM_ROWS=9600 SEAM_COLS=100000` + thread sweep | Westrick driver: `julia benchmarks/scripts/seam-westrick-scaling.jl` |
| **Game of Life** | `LIFE_ROWS=512 …` + `game-of-life.jl` | larger grid + `LIFE_SCENARIOS=both` | Stencil: registered Dagger `0.19.3` in project. |
| **Heat propagation** | `HEAT_*` small + `heat-propagation.jl` | `HEAT_STEPS=200`, `BENCH_RUNS=3` | Same stencil stack as game-of-life. |
| **pass@k** | `bash apps/pass-at-k-study/scripts/run_smoke_test.sh` | `cd apps/pass-at-k-study && bash scripts/run_full_study.sh gpt-4o-mini 5` | Smoke uses OpenAI key, **or** Ollama on `localhost:11434`, **or** a static JSONL fallback so phases 2–3 always run. Full needs a real model + API or Ollama. |

Exact `julia --project=… -e '…'` strings for each row are in **`AD_BENCHMARKS.md`** (authoritative) and are what the two orchestrator scripts call.

---

## Pass / fail (what to look for)

| After reduced run | Check |
|-------------------|--------|
| Cholesky | If a GPU was available: new directory under `benchmarks/results/gpu-cholesky/`. If skipped (`SKIP_CHOLESKY=1` or no GPU): expect `[skip] gpu-cholesky` — **not** a failure of other apps. |
| Barnes–Hut | `benchmarks/results/barnes-hut/…` with timing CSVs. |
| Seam | `benchmarks/results/seam-carving/…`. |
| Stencils | `benchmarks/results/game-of-life/…` and `…/heat-propagation/…`. |
| pass@k | `apps/pass-at-k-study/outputs/…` (smoke may use bundled JSONL; still runs evaluate/analyze). |

The orchestrators always print a final `=== … finished ===` line when the shell reaches the end.

---

## Environment knobs (all apps)

- **`run_smoke_all.sh`**: set **`SKIP_CHOLESKY=1`** to force-skip Cholesky on a **CPU-only** node while still running Barnes–Hut, seam, stencils, and pass@k.
- **Cholesky / CUDA.jl**: do **not** prepend `/usr/local/cuda` to `LD_LIBRARY_PATH` unless you know you need it; the scripts strip it. Set `CHOLESKY_KEEP_SYSTEM_CUDA_LD=1` to keep it.
- **pass@k**: for full runs, set **`OPENAI_API_KEY`** (or use Ollama) as in `AD_BENCHMARKS.md`. For a scoped full-style run, export `PASSK_TASK` / `PASSK_OUTPUT` before `run_paper_all.sh`.

---

## Relationship to the SC26 AD

The Artifact Description PDF cites branch **`SC26_AD_AE`**, tag **`sc26-ad-freeze`**, and this repo’s orchestrators. After `git checkout sc26-ad-freeze`, the SHAs in the PDF and `git rev-parse HEAD` must match.
