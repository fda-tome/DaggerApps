# Scripts

Automation utilities, batch runners, and helper scripts for benchmarks.

All app benchmark entry points live at `benchmarks/scripts/<app>.jl` and expose `run_benchmark()`. Script runners should call that entry point rather than re‑implementing logic.

## Scripts project

The `benchmarks/scripts/` folder is a lightweight Julia project that can be used to run benchmarks while developing the app implementation.

From the repo root (single‑node, threads‑based):

```bash
julia --project=benchmarks/scripts -t16 -e 'include("benchmarks/scripts/seam-carving.jl"); run_benchmark()'
julia --project=benchmarks/scripts -t16 -e 'include("benchmarks/scripts/game-of-life.jl"); run_benchmark()'
julia --project=benchmarks/scripts -t16 -e 'include("benchmarks/scripts/heat-propagation.jl"); run_benchmark()'
```

GPU Cholesky should use the **app** project (Dagger from `master`), not `benchmarks/scripts` alone:

```bash
julia --project=apps/gpu-cholesky -e 'using CUDA; using Dagger; include("benchmarks/scripts/gpu-cholesky.jl"); run_benchmark()'
```

If you want GPU runs, load any supported backend before running:

```julia
using CUDA # or AMDGPU / oneAPI / Metal
include("benchmarks/scripts/seam-carving.jl")
run_benchmark()
```

Game of Life and heat-propagation also expose GPU-only size sweeps (single thread count, varying matrix size):

```julia
using CUDA # or AMDGPU / oneAPI / Metal
include("benchmarks/scripts/game-of-life.jl")
run_gpu_size_sweep(sizes=[1024, 2048, 4096, 8192, 12288, 16384], variant=:gpu_dagger_stencil_wrap)

include("benchmarks/scripts/heat-propagation.jl")
run_gpu_size_sweep(sizes=[1024, 2048, 4096, 8192, 12288, 16384], variant=:gpu_dagger_stencil_pad)
```

## External project (terminal commands)

From any external Julia project:

```bash
julia --project=. -e 'using Pkg; Pkg.develop(path="/path/to/DaggerApps/benchmarks/scripts"); Pkg.instantiate()'
julia --project=. -e 'using DaggerAppsBenchmarks; run_seam_carving()'
julia --project=. -e 'using DaggerAppsBenchmarks; run_game_of_life()'
julia --project=. -e 'using DaggerAppsBenchmarks; run_heat_propagation()'
julia --project=. -e 'using CUDA, DaggerAppsBenchmarks; run_game_of_life_gpu_size_sweep(sizes=[1024,2048,4096,8192,12288,16384], variant=:gpu_dagger_stencil_wrap)' # CUDA/AMDGPU/oneAPI/Metal
julia --project=. -e 'using CUDA, DaggerAppsBenchmarks; run_heat_propagation_gpu_size_sweep(sizes=[1024,2048,4096,8192,12288,16384], variant=:gpu_dagger_stencil_pad)' # CUDA/AMDGPU/oneAPI/Metal
```

`run_gpu_cholesky()` from `DaggerAppsBenchmarks` loads the same script as the gpu-cholesky app; use an active project that resolves **Dagger** the way you intend (for the published `Manifest.toml`, use `--project=apps/gpu-cholesky` and `Pkg.develop` the `benchmarks/scripts` helper into that environment if you want the helper entry point).

If you want GPU runs, load a backend in the same session:

```bash
julia --project=. -e 'using CUDA, DaggerAppsBenchmarks; run_seam_carving()' # CUDA/AMDGPU/oneAPI/Metal all supported
```

## Notes

- Seam‑carving benchmarks run on a single Julia process (no `Distributed` workers). Control parallelism with `-t` / `JULIA_NUM_THREADS`.
- Game-of-life benchmarks also run on a single Julia process (no `Distributed` workers). Control parallelism with `-t` / `JULIA_NUM_THREADS`.
- Heat-propagation benchmarks also run on a single Julia process (no `Distributed` workers). Control parallelism with `-t` / `JULIA_NUM_THREADS`.
- Timing uses BenchmarkTools with a warmup run to avoid compilation time.
- Weak scaling uses `SEAM_WEAK_SCALE` (default `sqrt`) unless explicit `SEAM_WEAK_ROWS`/`SEAM_WEAK_COLS` are set.
- Weak scaling for game-of-life uses `LIFE_WEAK_SCALE` (default `sqrt`) unless explicit `LIFE_WEAK_ROWS`/`LIFE_WEAK_COLS` are set.
- Weak scaling for heat-propagation uses `HEAT_WEAK_SCALE` (default `sqrt`) unless explicit `HEAT_WEAK_ROWS`/`HEAT_WEAK_COLS` are set.
- GPU backend selection for game-of-life uses `LIFE_DEVICE=auto|cpu|cuda|amdgpu|oneapi|metal`.
- GPU backend selection for heat-propagation uses `HEAT_DEVICE=auto|cpu|cuda|amdgpu|oneapi|metal`.
- GPU-only size sweeps write `gpu_size_sweep_runs.csv`, `gpu_size_sweep_summary.csv`, and `gpu_size_sweep.png`.
- `DaggerAppsBenchmarks` exposes `run_seam_carving()`, `run_game_of_life()`, `run_heat_propagation()`, `run_gpu_cholesky()`, `run_game_of_life_gpu_size_sweep()`, and `run_heat_propagation_gpu_size_sweep()`.
