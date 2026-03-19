# Barnes–Hut

Distributed N-body simulation using the Barnes–Hut algorithm with Dagger.jl and Morton Z-curves. Designed for scalability to many nodes: partition once by Morton order (no repartition), no boundary conditions, and communication limited to multipole data.

## Algorithm

- **Morton Z-order**: 3D positions are encoded to a 63-bit Morton code (21 bits per dimension), then particles are sorted by code for spatial locality.
- **Distribution**: Particles are partitioned into contiguous Morton ranges; each Dagger worker owns one segment. No repartitioning during the run.
- **Local octree**: Each worker builds and keeps a local octree (spatial tree) for its segment only (Phase 3: no full-tree or coarse-tree broadcast).
- **Global multipole directory**: Once per timestep, each worker contributes its root multipole (center, mass, size); the directory is gathered and broadcast. Forces use the local tree plus these remote multipoles (θ criterion).
- **Integration**: Leapfrog-style update; no boundary conditions.

## Usage

Single process (serial path):

```bash
julia --project=apps/barnes-hut -e 'using Dagger; include("apps/barnes-hut/src/DaggerBarnesHut.jl"); using Main.DaggerBarnesHut; DaggerBarnesHut.bmark(10_000, 0.5)'
```

With Distributed workers (multi-node / multi-process):

```bash
julia --project=apps/barnes-hut -e '
using Distributed
addprocs(2)   # add workers before loading Dagger
using Dagger
include("apps/barnes-hut/src/DaggerBarnesHut.jl")
using Main.DaggerBarnesHut
DaggerBarnesHut.bmark(10_000, 0.5)
'
```

## Benchmark

From the repo root (DaggerApps):

```bash
julia --project=apps/barnes-hut benchmarks/scripts/barnes-hut.jl
```

Single run with a fixed number of workers (e.g. 32):

```bash
BARNES_NPROCS=32 julia --project=apps/barnes-hut benchmarks/scripts/barnes-hut.jl
```

**Strong and weak scaling experiment (1, 2, 4, 8, 16, 32 workers):**

```bash
BARNES_SCALING_EXPERIMENT=1 julia --project=apps/barnes-hut benchmarks/scripts/barnes-hut.jl
```

This launches a separate Julia process for each worker count, runs strong and weak scaling, then writes merged `scaling_strong.csv` and `scaling_weak.csv` under `benchmarks/results/barnes-hut/scaling_<timestamp>/`.

Environment variables (see `benchmarks/scripts/barnes-hut.jl`):

- `BENCH_WARMUP` – untimed `bmark` runs before strong and before weak to warm JIT (default: 1; set `0` to disable)
- `BENCH_RUNS` – number of timed runs per scenario (default: 3)
- `BENCH_BT_SAMPLES` – BenchmarkTools samples per timed run (default: 5)
- `BENCH_BT_EVALS` – BenchmarkTools evals per sample (default: 1)
- `BARNES_THETA` – opening angle θ (default: 0.5)
- `BARNES_N_STRONG` – N for strong scaling (default: 250000)
- `BARNES_BODIES_PER_PROC` – bodies per processor for weak scaling (default: 20000; e.g. 32 workers → N = 640000)

## Main API

- `bmark(N, theta; nsteps=1, dt=0.01)` – run benchmark: N particles, opening angle θ; uses serial or distributed path depending on `nprocs()` / `workers()`.
- `morton_encode(x, y, z)` – map 3D coords in [0,1] to Morton code.
- `ParticleSnapshot`, `random_particles(N)`, `run_timesteps` – for custom runs.

## Dependencies

- Dagger
- Distributed (stdlib)
- StaticArrays
