# DaggerApps

A collection of Dagger.jl application folders (apps), plus optional benchmark scaffolding. Each app is intended to be a self‑contained Julia project; benchmarks are opt‑in and live under `benchmarks/`.

**Exemplar / SC26 reviewer path:** see **[EXEMPLAR_QUICKSTART.md](EXEMPLAR_QUICKSTART.md)** (clone → verify SHA → `Pkg.instantiate` → `benchmarks/run_smoke_all.sh` → checkable outputs). License: [LICENSE](LICENSE) (MIT).

## SC26 AD/AE integration branch

Branch **`SC26_AD_AE`** merges the paper-pinned commits for Cholesky (NVIDIA/Intel
and AMD), Barnes–Hut, seam carving, stencil (`main` baseline), and the LLM
pass@k study. See **[BRANCHES_SC26.md](BRANCHES_SC26.md)** for full SHAs, merge
order, and entrypoint paths.

## Repo layout

```
DaggerApps/
├── apps/                      # Dagger applications (one folder per app)
│   ├── barnes-hut/            # Barnes–Hut N-body simulation (distributed, Morton Z-curve)
│   ├── game-of-life/          # Conway's Game of Life (stencil-based)
│   ├── gpu-cholesky/          # Multi-GPU DArray Cholesky (Dagger.jl pin in Project.toml)
│   ├── heat-propagation/      # 2D heat diffusion + animation
│   ├── pass-at-k-study/       # LLM pass@k comparative benchmark app
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

Quick start (pass@k study app):

```bash
julia --project=apps/pass-at-k-study -e 'using Pkg; Pkg.instantiate()'
bash apps/pass-at-k-study/scripts/run_smoke_test.sh
```

Smoke test **all supported runtimes** (dagger, iris, legate) in one shot:

```bash
bash apps/pass-at-k-study/scripts/smoke_all_runtimes.sh
```

(from `apps/pass-at-k-study`, or pass the full path). Set `LEGATE_PYTHON` if Legate is not at `../../../.venv/bin/python`.

Quick start (heat propagation animation):

```bash
julia --project=apps/heat-propagation apps/heat-propagation/scripts/animate.jl
```

Quick start (heat propagation benchmark):

```bash
julia --project=apps/heat-propagation -t16 -e 'include("benchmarks/scripts/heat-propagation.jl"); run_benchmark()'
```

Quick start (Barnes–Hut benchmark; optional `addprocs(N)` for distributed):

```bash
julia --project=apps/barnes-hut -e 'include("benchmarks/scripts/barnes-hut.jl"); run_benchmark()'
```

Quick start (GPU Cholesky benchmark; **four GPUs**; load a vendor package before `Dagger`):

```bash
julia --project=apps/gpu-cholesky -e 'using CUDA; using Dagger; include("benchmarks/scripts/gpu-cholesky.jl"); run_benchmark()'
```

Results go to `benchmarks/results/gpu-cholesky/<timestamp>/`. See `apps/gpu-cholesky/README.md` for env vars (`CHOLESKY_K_MIN`, `CHOLESKY_K_MAX`, etc.).

## External project usage

From any other Julia project:

```bash
julia --project=. -e 'using Pkg; Pkg.develop(path="/path/to/DaggerApps/benchmarks/scripts"); Pkg.instantiate()'
julia --project=. -e 'using DaggerAppsBenchmarks; run_seam_carving()'
julia --project=. -e 'using DaggerAppsBenchmarks; run_game_of_life()'
julia --project=. -e 'using DaggerAppsBenchmarks; run_heat_propagation()'
julia --project=. -e 'using DaggerAppsBenchmarks; run_barnes_hut()'
```

For **gpu-cholesky**, prefer `--project=apps/gpu-cholesky` so the environment pins **Dagger.jl** to GitHub `master` (see `apps/gpu-cholesky/Manifest.toml`). The `DaggerAppsBenchmarks` helper can call `run_gpu_cholesky()` but uses whatever Dagger version the active project resolves.

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
