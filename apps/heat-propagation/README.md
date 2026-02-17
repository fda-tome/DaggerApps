# Heat Propagation

This app provides `DaggerHeatPropagation` (`src/DaggerHeatPropagation.jl`) with a 2D heat diffusion example implemented both in serial and with Dagger stencils, plus an animation helper that writes a GIF.

## Contents

- `src/DaggerHeatPropagation.jl`: heat equation implementation + animation helper
- `scripts/animate.jl`: runnable script to generate a GIF
- `Project.toml`: Julia environment for the app

## Quick usage

First time only:

```bash
julia --project=apps/heat-propagation -e 'using Pkg; Pkg.instantiate()'
```

Run a short simulation and compare serial vs Dagger:

```bash
julia --project=apps/heat-propagation -e 'using DaggerHeatPropagation; plate=DaggerHeatPropagation.ambient_plate(128,128; ambient=0.0); DaggerHeatPropagation.add_hotspot!(plate; radius=8, temperature=1.0); serial, dagger, err = DaggerHeatPropagation.run_pipeline(plate; steps=200, alpha=0.2, block_h=32, block_w=32, boundary=:pad, pad_value=0.0); @show size(dagger) err'
```

Generate an animation (`heat_propagation.gif`):

```bash
julia --project=apps/heat-propagation apps/heat-propagation/scripts/animate.jl
```

Or override parameters via env vars:

```bash
HEAT_ROWS=256 HEAT_COLS=256 HEAT_STEPS=400 HEAT_BLOCK_H=64 HEAT_BLOCK_W=64 HEAT_SNAPSHOT_EVERY=4 HEAT_FPS=20 HEAT_GIF=heat_256.gif julia --project=apps/heat-propagation apps/heat-propagation/scripts/animate.jl
```

Run the benchmark entrypoint:

```bash
julia --project=apps/heat-propagation -t16 -e 'include("benchmarks/scripts/heat-propagation.jl"); run_benchmark()'
```

## Core functions

- `ambient_plate(rows, cols; ambient=0.0)`
- `add_hotspot!(plate; row, col, radius, temperature)`
- `add_gaussian_hotspot!(plate; row, col, sigma, amplitude)`
- `heat_propagate_cpu_serial(initial; steps, alpha, boundary, pad_value)`
- `heat_propagate_dagger_stencil(initial; steps, alpha, block_h, block_w, boundary, pad_value, return_darray)`
- `heat_propagate_dagger_history(initial; steps, alpha, block_h, block_w, boundary, pad_value, snapshot_every)`
- `save_heat_animation(path; rows, cols, steps, alpha, block_h, block_w, ambient, hotspot_temperature, hotspot_radius, boundary, pad_value, snapshot_every, fps, color)`

## Notes

- Uses an explicit 5-point stencil update; `alpha` must be in `(0, 0.25]` for stability.
- `boundary=:pad` applies a fixed outside temperature (`pad_value`), while `boundary=:wrap` uses periodic edges.
- The animation path stores snapshots from the Dagger stencil run and renders them with `Plots.jl`; use larger `HEAT_STEPS` for longer diffusion windows.
