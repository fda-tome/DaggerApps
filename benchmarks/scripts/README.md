# Scripts

Automation utilities, batch runners, and helper scripts for benchmarks.

All app benchmark entry points live at `benchmarks/scripts/<app>.jl` and expose `run_benchmark()`. Script runners should call that entry point rather than re‑implementing logic.

## Scripts project

The `benchmarks/scripts/` folder is a lightweight Julia project that can be used to run benchmarks while developing the app implementation.

From the repo root (single‑node, threads‑based):

```bash
julia --project=benchmarks/scripts -t16 -e 'include("benchmarks/scripts/seam-carving.jl"); run_benchmark()'
julia --project=benchmarks/scripts -t16 -e 'include("benchmarks/scripts/game-of-life.jl"); run_benchmark()'
```

If you want GPU runs, load a backend before running:

```julia
using CUDA # or AMDGPU / oneAPI / Metal
include("benchmarks/scripts/seam-carving.jl")
run_benchmark()
```

## External project (terminal commands)

From any external Julia project:

```bash
julia --project=. -e 'using Pkg; Pkg.develop(path="/path/to/DaggerApps/benchmarks/scripts"); Pkg.instantiate()'
julia --project=. -e 'using DaggerAppsBenchmarks; run_seam_carving()'
julia --project=. -e 'using DaggerAppsBenchmarks; run_game_of_life()'
```

If you want GPU runs, load a backend in the same session:

```bash
julia --project=. -e 'using CUDA, DaggerAppsBenchmarks; run_seam_carving()'
```

## Notes

- Seam‑carving benchmarks run on a single Julia process (no `Distributed` workers). Control parallelism with `-t` / `JULIA_NUM_THREADS`.
- Game-of-life benchmarks also run on a single Julia process (no `Distributed` workers). Control parallelism with `-t` / `JULIA_NUM_THREADS`.
- Timing uses BenchmarkTools with a warmup run to avoid compilation time.
- Weak scaling uses `SEAM_WEAK_SCALE` (default `sqrt`) unless explicit `SEAM_WEAK_ROWS`/`SEAM_WEAK_COLS` are set.
- Weak scaling for game-of-life uses `LIFE_WEAK_SCALE` (default `sqrt`) unless explicit `LIFE_WEAK_ROWS`/`LIFE_WEAK_COLS` are set.
- `DaggerAppsBenchmarks` exposes `run_seam_carving()` and `run_game_of_life()`.
