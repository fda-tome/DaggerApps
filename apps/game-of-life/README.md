# Game of Life

This app provides the `DaggerGameOfLife` module (`src/DaggerGameOfLife.jl`) with a serial baseline and a Dagger stencil implementation of Conway's Game of Life. Functions are not exported, so call them with the `DaggerGameOfLife.` prefix.

## Contents

- `src/DaggerGameOfLife.jl`: implementation (serial + Dagger stencil variants)
- `Project.toml`: Julia environment for the app

## Quick usage

First time only:

```bash
julia --project=apps/game-of-life -e 'using Pkg; Pkg.instantiate()'
```

Run a small simulation:

```bash
julia --project=apps/game-of-life -e 'using DaggerGameOfLife; world=DaggerGameOfLife.random_world(128,128; density=0.2); out=DaggerGameOfLife.game_of_life_dagger_stencil(world; steps=50, block_h=32, block_w=32); @show size(out) DaggerGameOfLife.alive_count(out)'
```

## Available variants

- Serial: `game_of_life_cpu_serial(initial; steps, boundary, pad_value)`
- Dagger stencil: `game_of_life_dagger_stencil(initial; steps, block_h, block_w, boundary, pad_value, return_darray)`

Boundary behavior:

- `boundary=:wrap` (periodic edges, classic toroidal world)
- `boundary=:pad` (outside grid treated as `pad_value`, default `false`)

## Helpers

- `random_world(rows, cols; density=0.25)`
- `seed_glider!(world; row=2, col=2)`
- `seed_blinker!(world; row=2, col=2, horizontal=true)`
- `alive_count(world)`

## Notes

- The stencil path uses `Dagger.spawn_datadeps()` + `@stencil` + `@neighbors`, matching Dagger's stencil execution model.
- CPU parallelism comes from Julia threads (`-t` / `JULIA_NUM_THREADS`) and Dagger scheduling over chunked `DArray`s.
