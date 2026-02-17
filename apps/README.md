# Apps

Apps (Dagger.jl applications) live here. Each app directory is intended to be a self‑contained Julia project with its own `Project.toml` (and optionally a `Manifest.toml`).

## Usage

Instantiate an app environment:

```bash
julia --project=apps/<app> -e 'using Pkg; Pkg.instantiate()'
```

For app‑specific entry points and examples, see each app’s `README.md`.

## Current apps

- `barnes-hut/`: Barnes–Hut N‑body simulation (placeholder).
- `game-of-life/`: Conway's Game of Life (serial + Dagger stencil implementation).
- `seam-carving/`: Content‑aware image resizing (implemented).
