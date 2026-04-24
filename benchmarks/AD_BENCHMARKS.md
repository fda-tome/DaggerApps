# AD benchmark presets (reduced vs paper)

**Reviewer one-pager (two shell commands, per-app table, pass/fail):** see **`REVIEWER_PATHS.md`** in this directory.  
**Exemplar clone + instantiate:** **`EXEMPLAR_QUICKSTART.md`** in the repository root.

---

All case studies use **one canonical Julia/bash entrypoint** per workload. The **paper** tier matches the Artifact Description defaults; the **reduced / AE** tier changes **only environment variables and numeric flags** (sizes, trials, workers, samples). **Dagger.jl** is **not** a single tag for every app: Cholesky, Barnes–Hut, seam, pass@$k$, and the benchmark helper projects use `Project.toml` **`[sources]`** with `rev =` the **full commit SHA** (same tree as the `fda/sc26-ad` line of development, e.g. `d9bf3ca…`); the two **stencil** apps pin **registered** `Dagger = "0.19.3"` in **`[compat]`** (see the SC26 Artifact Description PDF / its LaTeX source and each `Project.toml` / `Manifest.toml`). After pulling, `Pkg.instantiate()`.

**All apps, orchestrated (recommended):** from the repo root, `bash benchmarks/run_smoke_all.sh` (reduced) and `bash benchmarks/run_paper_all.sh` (full) invoke the same canonical `include("benchmarks/scripts/…")` and pass@k scripts described below.

Run commands from the **`DaggerApps` repository root** (the directory that contains `apps/` and `benchmarks/`).

---

## 1. GPU Cholesky (`run_benchmark()`)

**Canonical invocation (CUDA example):**

```bash
julia --project=apps/gpu-cholesky -e 'using CUDA; using Dagger; include("benchmarks/scripts/gpu-cholesky.jl"); run_benchmark()'
```

AMD / Intel / Apple: load `AMDGPU`, `oneAPI`, or `Metal` instead of `CUDA` before `Dagger`.

| Tier | Parameters (env) |
|------|------------------|
| **Reduced** | `CHOLESKY_NUM_GPUS=1` (or `2`/`3` if that matches visible devices), `CHOLESKY_NS=1024,2048`, `CHOLESKY_TRIALS=1`, `CHOLESKY_WARMUP=0`, `CHOLESKY_VENDOR=0` optional. With one GPU, default `:rl_la` is remapped to `:rl` (set `CHOLESKY_FORCE_RL_LA=1` to override). **Do not** put `/usr/local/cuda/lib64` on `LD_LIBRARY_PATH` when using CUDA.jl — it can crash the process; `run_smoke_all.sh` / `run_paper_all.sh` strip it unless `CHOLESKY_KEEP_SYSTEM_CUDA_LD=1`. |
| **Paper** | Defaults in `gpu-cholesky.jl` / AD (`CHOLESKY_NUM_GPUS=4` on 4-GPU nodes, `CHOLESKY_K_MIN`/`CHOLESKY_K_MAX` sweep, `CHOLESKY_TRIALS=5`, etc.) |

**Full sweep (same driver stack, optional):** after `cd` to repo root, `julia --project=apps/gpu-cholesky benchmarks/scripts/gpu-cholesky-sweep.jl` — uses the same `CHOLESKY_NUM_GPUS` logic as `gpu-cholesky.jl` for precompile.

---

## 2. Barnes–Hut (`run_benchmark()`)

**Canonical:**

```bash
export BARNES_NPROCS=4
julia --project=apps/barnes-hut -e 'using Dagger; include("benchmarks/scripts/barnes-hut.jl"); run_benchmark()'
```

| Tier | Parameters |
|------|------------|
| **Reduced** | `BARNES_NPROCS=4`, `BARNES_N_STRONG=50000`, `BENCH_RUNS=1`, `BENCH_WARMUP=0`, `BENCH_BT_SAMPLES=1` |
| **Paper** | `BARNES_NPROCS` per allocation, `BARNES_N_STRONG=250000`, defaults for `BENCH_*` |

---

## 3. Seam carving (`run_benchmark()`)

**Canonical:**

```bash
julia --threads=auto --project=apps/seam-carving -e 'using Dagger; include("benchmarks/scripts/seam-carving.jl"); run_benchmark()'
```

| Tier | Parameters |
|------|------------|
| **Reduced** | `SEAM_ROWS=256`, `SEAM_COLS=256`, `BENCH_RUNS=1`, `SEAM_K=1`, `SEAM_SCENARIOS=strong` (or `both` if time allows) |
| **Paper** | `SEAM_ROWS=9600`, `SEAM_COLS=100000` (Aurora single-node Fig. `fig:seam-strong`), `BENCH_RUNS=3`, `SEAM_THREAD_SWEEP=1,2,4,8,16,32,52`, `SEAM_SCENARIOS=both`, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` |

Westrick thread sweep driver (same seam app project):

```bash
julia --project=apps/seam-carving benchmarks/scripts/seam-westrick-scaling.jl
```

Tune `SEAM_THREAD_SWEEP`, `SEAM_SCENARIOS`, etc. via env only.

**Orchestrators:** `benchmarks/run_smoke_all.sh` and `run_paper_all.sh` invoke this driver after `run_benchmark()` for seam (shorter default `SEAM_THREAD_SWEEP` on reduced tier; full sweep on paper). Set `OMP_NUM_THREADS=1` (and BLAS thread caps) as in `apps/seam-carving/README.md` if your site needs it—the scripts export `OMP_NUM_THREADS=1` by default for the Westrick block.

---

## 4. Game of Life stencil (`run_benchmark()`)

**Canonical:**

```bash
julia --project=apps/game-of-life -e 'using Dagger; include("benchmarks/scripts/game-of-life.jl"); run_benchmark()'
```

| Tier | Parameters |
|------|------------|
| **Reduced** | `LIFE_ROWS=512`, `LIFE_COLS=512`, `LIFE_STEPS=20`, `BENCH_RUNS=1`, `LIFE_SCENARIOS=strong` |
| **Paper** | Defaults / AD sizes, `LIFE_STEPS=100`, `BENCH_RUNS=3`, `LIFE_SCENARIOS=both` |

---

## 5. Heat propagation stencil (`run_benchmark()`)

**Canonical:**

```bash
julia --project=apps/heat-propagation -e 'using Dagger; include("benchmarks/scripts/heat-propagation.jl"); run_benchmark()'
```

| Tier | Parameters |
|------|------------|
| **Reduced** | `HEAT_ROWS=512`, `HEAT_COLS=512`, `HEAT_STEPS=50`, `BENCH_RUNS=1`, `HEAT_SCENARIOS=strong` |
| **Paper** | AD defaults, `HEAT_STEPS=200`, `BENCH_RUNS=3`, `HEAT_SCENARIOS=both` |

---

## 6. pass@k study (generate → evaluate → analyze)

**Canonical pipeline:** `apps/pass-at-k-study/scripts/run_full_study.sh` (all tiers).

```bash
cd apps/pass-at-k-study
bash scripts/run_full_study.sh gpt-4o-mini 5
```

| Tier | Parameters |
|------|------------|
| **Reduced** | `PASSK_TASK=l1_basic_spawn`, `PASSK_OUTPUT=outputs/generated/smoke.jsonl`, second arg `2` for `--n-samples`; optional `API_BASE` for Ollama |
| **Paper** | All tasks (no `PASSK_TASK`), default output naming, `--n-samples` 5 (or AD value). To scope the paper orchestrator to one task, export `PASSK_TASK` / `PASSK_OUTPUT` before `run_paper_all.sh` (same `run_full_study.sh` entrypoint). |

Optional env for **`run_full_study.sh`** (same script for all tiers):

| Variable | Role |
|----------|------|
| `PASSK_TASK` | `--task` filter for `generate.jl` |
| `PASSK_OUTPUT` | `--output` path for `generate.jl` |
| `PASSK_SKIP_GENERATE=1` | skip phase 1 |
| `PASSK_GENERATED` | path to existing JSONL when skipping generate |
| `PASSK_API_BASE`, `PASSK_API_KEY` | override `API_BASE` / API key for the generator |

With `PASSK_SKIP_GENERATE=1` and `PASSK_GENERATED=…`, phases 2–3 reuse the same script after a pre-built JSONL.

---

## Dagger pin

Most case-study `apps/*` (Cholesky, Barnes–Hut, seam, pass@$k$) use a **git SHA** in `[sources]`, e.g.:

```toml
[sources]
Dagger = {url = "https://github.com/JuliaParallel/Dagger.jl", rev = "d9bf3ca494bdbca8e0e7e52501010da20f481bd9"}
```

**Stencil** apps (`game-of-life`, `heat-propagation`) use **`[compat] Dagger = "0.19.3"`** (registry) instead. Experimental `apps/multi-gpu-*` may still use `rev = "fda/sc26-ad"`; check that project’s `Project.toml`. Run `Pkg.instantiate()` (and `Pkg.resolve()` if you edit dependencies) after pulling.
