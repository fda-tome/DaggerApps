# DaggerApps

A collection of Dagger.jl application folders (apps), plus optional benchmark scaffolding. Each app is intended to be a self‑contained Julia project; benchmarks are opt‑in and live under `benchmarks/`.

## Repo layout

```
DaggerApps/
├── apps/                      # Dagger applications (one folder per app)
│   ├── barnes-hut/            # Barnes–Hut N-body simulation (placeholder)
│   ├── game-of-life/          # Conway's Game of Life (stencil-based)
│   ├── heat-propagation/      # 2D heat diffusion + animation
│   └── seam-carving/          # Content-aware image resizing (seam carving)
└── benchmarks/                # Optional benchmark suite for the apps
```

## Quick start (seam‑carving benchmark)

From the repo root, single‑node (no Distributed workers):

```bash
julia --project=apps/seam-carving -t16 -e 'include("benchmarks/scripts/seam-carving.jl"); run_benchmark()'
```

Results are written to `benchmarks/results/seam-carving/<timestamp>/`.

Quick start (game-of-life benchmark):

```bash
julia --project=apps/game-of-life -t16 -e 'include("benchmarks/scripts/game-of-life.jl"); run_benchmark()'
```

Results are written to `benchmarks/results/game-of-life/<timestamp>/`.

Quick start (heat propagation animation):

```bash
julia --project=apps/heat-propagation apps/heat-propagation/scripts/animate.jl
```

Quick start (heat propagation benchmark):

```bash
julia --project=apps/heat-propagation -t16 -e 'include("benchmarks/scripts/heat-propagation.jl"); run_benchmark()'
```

## External project usage

From any other Julia project:

```bash
julia --project=. -e 'using Pkg; Pkg.develop(path="/path/to/DaggerApps/benchmarks/scripts"); Pkg.instantiate()'
julia --project=. -e 'using DaggerAppsBenchmarks; run_seam_carving()'
julia --project=. -e 'using DaggerAppsBenchmarks; run_game_of_life()'
julia --project=. -e 'using DaggerAppsBenchmarks; run_heat_propagation()'
```

This uses the `DaggerAppsBenchmarks` helper package (defined in `benchmarks/scripts/`).

## GPU runs

Load a backend in the same Julia session before running the seam‑carving benchmark:

```julia
using CUDA # or AMDGPU / oneAPI / Metal
using DaggerAppsBenchmarks
run_seam_carving()
```

You can also set `SEAM_DEVICE=cuda|amdgpu|oneapi|metal` to select a backend explicitly.

## Contributing

- Add new apps under `apps/<name>/` and include a short `README.md` plus a Julia project (`Project.toml`).
- Keep apps runnable by default; document any cluster/GPU/MPI requirements.

## Related resources

- https://juliaparallel.org/Dagger.jl/stable/
- https://docs.julialang.org/en/v1/manual/parallel-computing/
- https://github.com/JuliaParallel

---

Note: apps and benchmarks evolve; for reproducible runs, rely on each app’s `Manifest.toml` when present.
