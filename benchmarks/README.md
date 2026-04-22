# Benchmarks

Optional benchmark suite for the apps in `apps/`.

**AD-aligned presets:** [AD_BENCHMARKS.md](AD_BENCHMARKS.md) documents canonical commands per case study; reduced vs full tiers differ by **environment variables only**. Orchestrators: `run_smoke_all.sh`, `run_paper_all.sh`.

## Entry points (per app)

For every app folder `apps/<app>/`, there is a matching benchmark script:

- `benchmarks/scripts/<app>.jl`

Each script defines a single entry point:

- `run_benchmark()`

Calling `run_benchmark()` runs:

- **Strong scaling**: fixed problem size (measure performance under the current resource configuration, e.g. Julia threads / Dagger processors).
- **Weak scaling**: increase problem size with available resources (the exact scaling rule is benchmark-specific).

To generate a scaling curve, rerun the same benchmark under different resource allocations (threads/workers/MPI ranks) and aggregate the CSV outputs.

## Data and results (per app)

- `benchmarks/data/<app>/`: input datasets / test assets
- `benchmarks/results/<app>/`: outputs (CSVs, logs, plots), typically in timestamped subfolders

Keep large binaries out of git when possible.

## Running

From the repo root:

```bash
# Seam carving (single-node, threads-based)
julia --project=apps/seam-carving -t16 -e 'include("benchmarks/scripts/seam-carving.jl"); run_benchmark()'

# Game of Life (single-node, threads-based)
julia --project=apps/game-of-life -t16 -e 'include("benchmarks/scripts/game-of-life.jl"); run_benchmark()'

# Heat propagation (single-node, threads-based)
julia --project=apps/heat-propagation -t16 -e 'include("benchmarks/scripts/heat-propagation.jl"); run_benchmark()'

# GPU Cholesky (1–4 GPUs via CHOLESKY_NUM_GPUS; load CUDA/AMDGPU/oneAPI/Metal before Dagger)
julia --project=apps/gpu-cholesky -e 'using CUDA; using Dagger; include("benchmarks/scripts/gpu-cholesky.jl"); run_benchmark()'
# AMD / ROCm (e.g. MI300-class): use AMDGPU first, optional CHOLESKY_DEVICE=amdgpu
julia --project=apps/gpu-cholesky -e 'using AMDGPU; using Dagger; include("benchmarks/scripts/gpu-cholesky.jl"); run_benchmark()'
```

If you want to run from an external project, develop the benchmarks package once:

```bash
julia --project=. -e 'using Pkg; Pkg.develop(path="/path/to/DaggerApps/benchmarks/scripts"); Pkg.instantiate()'
julia --project=. -e 'using DaggerAppsBenchmarks; run_seam_carving()'
julia --project=. -e 'using DaggerAppsBenchmarks; run_game_of_life()'
julia --project=. -e 'using DaggerAppsBenchmarks; run_heat_propagation()'
```

Notes for seam‑carving:

- Timing uses BenchmarkTools with a warmup run to avoid compilation time.
- Strong scaling keeps `SEAM_ROWS`/`SEAM_COLS` fixed.
- Weak scaling scales rows/cols with threads via `SEAM_WEAK_SCALE` (or explicit `SEAM_WEAK_ROWS`/`SEAM_WEAK_COLS`).

Notes for game-of-life:

- Timing uses BenchmarkTools with a warmup run to avoid compilation time.
- Strong scaling keeps `LIFE_ROWS`/`LIFE_COLS` fixed.
- Weak scaling scales rows/cols with threads via `LIFE_WEAK_SCALE` (or explicit `LIFE_WEAK_ROWS`/`LIFE_WEAK_COLS`).
- GPU variants can be enabled with any loaded backend (`CUDA`, `AMDGPU`, `oneAPI`, `Metal`) and selected via `LIFE_DEVICE`.
- GPU-only matrix-size sweeps are available with `run_gpu_size_sweep(...)` in `benchmarks/scripts/game-of-life.jl`.

Notes for heat-propagation:

- Timing uses BenchmarkTools with a warmup run to avoid compilation time.
- Strong scaling keeps `HEAT_ROWS`/`HEAT_COLS` fixed.
- Weak scaling scales rows/cols with threads via `HEAT_WEAK_SCALE` (or explicit `HEAT_WEAK_ROWS`/`HEAT_WEAK_COLS`).
- GPU variants can be enabled with any loaded backend (`CUDA`, `AMDGPU`, `oneAPI`, `Metal`) and selected via `HEAT_DEVICE`.
- GPU-only matrix-size sweeps are available with `run_gpu_size_sweep(...)` in `benchmarks/scripts/heat-propagation.jl`.

Notes for gpu-cholesky:

- Uses **up to four** visible GPUs of one vendor (`CHOLESKY_NUM_GPUS`, default 4); tile `assignment=` generalizes the paper 2×2 block-cyclic layout (degenerate assignment on one GPU).
- Default matrix sizes are `N = 2^k` from `k = 10` through `18` (override with `CHOLESKY_K_MIN` / `CHOLESKY_K_MAX` or `CHOLESKY_NS`).
- The script also records a **vendor baseline** (same loaded backend as Dagger: CUDA / AMDGPU / oneAPI / Metal): single-GPU `LinearAlgebra.cholesky!` on a dense GPU matrix matching the same SPD problem (`CHOLESKY_VENDOR`, `CHOLESKY_VENDOR_DEVICE`). CSV columns `dagger_*` vs `vendor_*` compare the two; vendor timing includes a per-trial `copyto!` refresh before `cholesky!`.
- Optional **`CHOLESKY_PERF_LOG=1`**: Dagger TimespanLogging summaries per `(N, block_size)` as NDJSON (`perf_dagger.jsonl`; see `CHOLESKY_PERF_SCOPE`, `CHOLESKY_PERF_LOG_PATH`).
- **`CHOLESKY_ALGO=rl_la`** (default): algorithm variant (`rl`, `rl_la`, `ll`); comma-separated list sweeps all in one run. `rl_la` adds processor pinning and lookahead spawn ordering; `ll` is a left-looking GEMM-based variant with pinning.
- **`CHOLESKY_BLOCKS`** / **`CHOLESKY_INPLACE`**: tile-size sweep and in-place `cholesky!` path (see script header / app README).
- The app environment pins **Dagger.jl** to GitHub **`fda/sc26-ad`** via `Project.toml` `[sources]` and `Manifest.toml`.
- Prefer `CHOLESKY_ELTYPE=Float32` for large `N` (memory).

## Folder layout

- `benchmarks/data/`
- `benchmarks/results/`
- `benchmarks/scripts/`
