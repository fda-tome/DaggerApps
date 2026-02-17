# Benchmarks

Optional benchmark suite for the apps in `apps/`.

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
```

If you want to run from an external project, develop the benchmarks package once:

```bash
julia --project=. -e 'using Pkg; Pkg.develop(path="/path/to/DaggerApps/benchmarks/scripts"); Pkg.instantiate()'
julia --project=. -e 'using DaggerAppsBenchmarks; run_seam_carving()'
julia --project=. -e 'using DaggerAppsBenchmarks; run_game_of_life()'
```

Notes for seam‑carving:

- Timing uses BenchmarkTools with a warmup run to avoid compilation time.
- Strong scaling keeps `SEAM_ROWS`/`SEAM_COLS` fixed.
- Weak scaling scales rows/cols with threads via `SEAM_WEAK_SCALE` (or explicit `SEAM_WEAK_ROWS`/`SEAM_WEAK_COLS`).

Notes for game-of-life:

- Timing uses BenchmarkTools with a warmup run to avoid compilation time.
- Strong scaling keeps `LIFE_ROWS`/`LIFE_COLS` fixed.
- Weak scaling scales rows/cols with threads via `LIFE_WEAK_SCALE` (or explicit `LIFE_WEAK_ROWS`/`LIFE_WEAK_COLS`).

## Folder layout

- `benchmarks/data/`
- `benchmarks/results/`
- `benchmarks/scripts/`
