# Apps

Apps (Dagger.jl applications) live here. Each app directory is intended to be a self‑contained Julia project with its own `Project.toml` (and optionally a `Manifest.toml`).

## Usage

Instantiate an app environment:

```bash
julia --project=apps/<app> -e 'using Pkg; Pkg.instantiate()'
```

For app‑specific entry points and examples, see each app’s `README.md`.

## Current apps

- `barnes-hut/`: Barnes–Hut N‑body simulation (distributed, Morton Z-curve, Dagger).
- `gpu-cholesky/`: Multi-GPU `DArray` Cholesky benchmark (Dagger `fda/sc26-ad`, 1–4 GPUs via env; vendor-agnostic GPU backends).
- `game-of-life/`: Conway's Game of Life (serial + Dagger stencil implementation).
- `heat-propagation/`: 2D heat diffusion with Dagger stencils and GIF animation helper.
- `pass-at-k-study/`: LLM pass@k comparative study app (Dagger, Iris, Legate).
- `seam-carving/`: Content‑aware image resizing (implemented).
