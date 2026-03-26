# Seam Carving

This app provides the `DaggerSeamCarving` module (`src/DaggerSeamCarving.jl`) with CPU and GPU seam‑carving variants. Functions are not exported, so call them with the `DaggerSeamCarving.` prefix.

## Contents

- `src/DaggerSeamCarving.jl`: seam‑carving implementation (serial + Dagger variants)
- `Project.toml` / `Manifest.toml`: Julia environment for the app

## Quick usage

First time only:

```bash
julia --project=apps/seam-carving -e 'using Pkg; Pkg.instantiate()'
```

```bash
julia --project=apps/seam-carving -e 'using DaggerSeamCarving; img=rand(Float32, 512, 512); DaggerSeamCarving.seam_carve_cpu_serial(img; k=1)'
```

Available variants (all require an `AbstractMatrix` / `AbstractArray`):

- CPU: `seam_carve_cpu_serial`, `seam_carve_cpu_dagger`, `seam_carve_cpu_dagger_tiled`, `seam_carve_cpu_dagger_wavefront`, `seam_carve_cpu_dagger_tileoverlap`, `seam_carve_cpu_dagger_triangles`, `seam_carve_cpu_dagger_westrick`
- GPU: `seam_carve_gpu_dagger`, `seam_carve_gpu_dagger_device` (expects GPU arrays)

GPU example:

```bash
julia --project=apps/seam-carving -e 'using CUDA, DaggerSeamCarving; img=CUDA.CuArray(rand(Float32, 512, 512)); DaggerSeamCarving.seam_carve_gpu_dagger(img; k=1)'
```

## Parallelism flavors

Seam carving has a mix of *algorithmic* dependencies (some steps must happen in order) and *implementation* choices about how to expose parallel work. All variants here are **single-node** (one Julia process); CPU parallelism comes from Julia threads (`-t` / `JULIA_NUM_THREADS`).

- **Pure serial baseline**: `seam_carve_cpu_serial` uses fully serial kernels (`energy_cpu_serial`, `cumulative_energy_cpu_serial`, `remove_seam_serial`). This is your reference for correctness and “no parallelism” overhead.
- **CPU loop threading**: most non-serial CPU helpers use `Threads.@threads` internally (e.g. `energy_cpu`, `cumulative_energy_cpu`, `remove_seam`). This is data-parallelism *within* each stage.
- **Dagger task graphs (CPU)**: the `seam_carve_cpu_dagger*` family uses Dagger tasks (`Dagger.@spawn`) to express higher-level parallel work and dependencies.
  - `seam_carve_cpu_dagger`: coarse pipeline over stages (energy → DP → backtrack → remove), where each stage can itself be threaded.
  - `seam_carve_cpu_dagger_tiled`: embarrassingly-parallel tiling for energy and seam-removal; DP/backtrack are still computed in a single step.
  - `seam_carve_cpu_dagger_wavefront`: a tiled DP with a *wavefront* dependency pattern (each DP tile depends on tiles “above” it). By default, energy and seam removal use **serial** loops so parallelism comes from **Dagger DP tasks only** (avoids `Threads.@threads` competing with Dagger). Set `SEAM_CPU_NESTED_THREADS=1` to restore threaded energy/removal for A/B.
  - **DP width:** with horizontal tile width `tile_w`, the wavefront uses `ntx = ceil(W / tile_w)` Dagger chunks per row — roughly **`ntx` concurrent DP tasks** per row wave, not `JULIA_NUM_THREADS`. The outer `tile_h` only groups row indices in the driver loop; it does **not** reduce how many Dagger tasks run per row.
  - **Hybrid (many cores):** use a **wider** `tile_w` (smaller `ntx`) to cut Dagger overhead, and set **`SEAM_DP_INNER_THREADS`** so each chunk parallelizes its `x` range with a **capped** `Threads.@threads` split (aim for `ntx × inner ≈` thread count). Optional **`SEAM_DP_INNER_MIN_SPAN`** and **`SEAM_DP_INNER_MIN_CHUNK`** avoid over-parallelizing narrow slices.
  - `seam_carve_cpu_dagger_tileoverlap`: overlaps tiled energy with the DP wavefront to increase concurrency.
  - `seam_carve_cpu_dagger_triangles`: same default as wavefront (serial energy/removal + Dagger DP); optional `SEAM_CPU_NESTED_THREADS=1` for legacy behavior. Inner DP threading uses the same `SEAM_DP_INNER_THREADS` / `SEAM_DP_INNER_MIN_SPAN` / `SEAM_DP_INNER_MIN_CHUNK` knobs via `dp_row_range!`.
  - `seam_carve_cpu_dagger_westrick`: **Westrick triangular-blocked strip DP** (two Dagger phases per strip: all upper triangles, then all lower triangles), following the schedule in MPL [`examples/src/seam-carve/SCI.sml`](https://github.com/MPLLang/mpl/blob/main/examples/src/seam-carve/SCI.sml) and described in [Parallel Seam Carving](https://shwestrick.github.io/2020/07/29/seam-carve.html). Energy uses **Dagger-tiled** writes with **serial** inner loops (`energy_cpu_dagger_tiled_serial`); removal uses **Dagger** row blocks with **serial** copies (`remove_seam_dagger_tiled_serial`). **No** nested `Threads.@threads` in the hot path—parallelism is Dagger tasks only. Dagger **wave** size: unless `SEAM_WAVE_SIZE` is set, each **phase** uses a wave at least as large as its task count (avoids serializing many fetches). Helpers `westrick_dp_matches_serial` / `westrick_dagger_dp_matches_serial` compare against `cumulative_energy_cpu_serial` on the same `E`.
- **GPU kernel parallelism (KernelAbstractions)**: GPU variants use `@kernel` definitions for energy/DP/remove and rely on a loaded backend (CUDA/AMDGPU/oneAPI/Metal).
  - `seam_carve_gpu_dagger`: runs GPU kernels but backtracks the seam on the CPU (host/device copies).
  - `seam_carve_gpu_dagger_device`: keeps seam backtracking on-device (`find_seam_gpu_device`) to avoid host round-trips.

Note: some variants combine Dagger task parallelism *and* `Threads.@threads` inside tasks; when exploring CPU performance, be mindful of potential nested parallelism/oversubscription.

## Benchmarks (single‑node)

The seam‑carving benchmark runs on a single Julia process (no `Distributed` workers). Use threads via `-t` or `JULIA_NUM_THREADS`.

From the repo root:

```bash
julia --project=apps/seam-carving -t16 -e 'include("benchmarks/scripts/seam-carving.jl"); run_benchmark()'
```

Results are written under `benchmarks/results/seam-carving/<timestamp>/`.

### Configuration

Benchmarks are driven by environment variables (timed with BenchmarkTools after a warmup run):

- `BENCH_RUNS` (default: 3)
- `SEAM_ROWS` / `SEAM_COLS` (default: 512x512)
- `SEAM_K` (default: 1)
- `SEAM_TILE_H` / `SEAM_TILE_W` (default: 150x150)
- `SEAM_VARIANTS` (default: all; comma/space‑separated)
- `SEAM_GPU` (default: 1; set to 0 to skip GPU variants)
- `SEAM_DEVICE` (default: auto; cpu|cuda|amdgpu|oneapi|metal)
- `SEAM_WEAK_SCALE` (default: sqrt; options: sqrt|linear|<float>)
- `SEAM_WEAK_ROWS` / `SEAM_WEAK_COLS` (override weak dimensions)
- `SEAM_WAVE_SIZE` (optional): hard cap on Dagger tasks per **fetch wave** for wavefront/triangles and for westrick when set. Default (unset) scales with `-t` via `clamp(4*t,128,2048)`. **Westrick + energy/remove (unset only):** the implementation uses `max(default_wave, num_tasks_in_phase)` so a phase is **not** split into multiple sequential waves when `128 < num_tasks` (e.g. wide images at `-t1` used to serialize 131 DP tasks into two waves). If you set `SEAM_WAVE_SIZE`, that value is a **strict** cap (no expansion)—use it to limit peak concurrency / memory.
- `SEAM_PIN_THREADS` (optional, default off): if set, calls [`ThreadPinning.jl`](https://github.com/JuliaParallel/ThreadPinning.jl) via `seam_configure_thread_pinning!()` so Julia worker threads are **affined** to CPUs and migrate less. Typical: `SEAM_PIN_THREADS=compact` or `cores`. Use together with **`numactl`** / your job’s CPU set. The benchmark script invokes this automatically when the variable is set.
- `SEAM_WESTRICK_BLOCK_WIDTH` (optional, default `80`): **even** integer, MPL `block-width`. Triangle height is `block_width ÷ 2`; strip advance is `block_height + 1` rows per global sync pair. Wider blocks raise per-task grain (~O(block_width²) cells in an upper triangle); tune with image width and thread count (the blog targets ~1k–2k cells per triangle for amortized parallelism overhead).
- `SEAM_WESTRICK_BLOCK_WIDTH_AUTO` (`0`/`1`): when `1`, choose an **even** `block_width` automatically from image width and `-t` (fewer vertical strips when possible, capped by `SEAM_WESTRICK_BLOCK_WIDTH_AUTO_MAX`, default 512).
- `SEAM_WESTRICK_BLOCK_WIDTH_SCHED_THREADS` (optional): positive integer **`S`**. When set with auto width, strip geometry uses **`S`** instead of the live `julia -t` when calling `_westrick_auto_sched_T` (so **`block_width` / strip count stay constant** across a 1…`S` strong-scaling sweep while energy/remove still use all threads). **Precedence with `SEAM_WESTRICK_BLOCK_WIDTH_AUTO_TASK_TARGET`:** effective strip scheduler is **`Tsched = min(S, cap)`** when both are set (otherwise `min(-t, cap)` or `-t` as before). Set **`S` to the maximum thread count in your sweep** (e.g. 52). Some Julia threads may idle during Westrick DP when `-t > num_upper_column_blocks`.
- `SEAM_WESTRICK_BLOCK_WIDTH_AUTO_TASK_TARGET` (optional): positive integer cap **`cap`**. Strip geometry uses `min` of the thread count chosen above and **`cap`** (`min(-t, cap)` when `SCHED_THREADS` is unset; `min(S, cap)` when it is set)—**wider** blocks, **fewer** global DP sync rounds at high core counts, at the cost of idle threads in DP. Example: logs at `10400×2400` showed `-t52` narrowing to `block_width≈202` (24 strips, large DP allocations) without these knobs.
- `SEAM_CPU_NESTED_THREADS` (default `0`): for **wavefront** and **triangles** only, `0` = serial `energy_cpu_serial` / `remove_seam_serial`, `1` = threaded `energy_cpu` / `remove_seam`.
- `SEAM_MIN_DP_TILE_W` (optional): if set, DP horizontal tile width is at least this value (capped by image width) for wavefront/triangles—useful when per-task overhead is large (also **reduces** `ntx` and Dagger concurrency).
- `SEAM_DP_INNER_THREADS` (optional, default `1`): max Julia threads for the **inner** `x` loop inside one DP row chunk (`dp_row_range!`, wavefront + triangles). Use with **wider** `tile_w` for a capped hybrid (`ntx × inner ≈` hardware threads). Capped at `JULIA_NUM_THREADS`.
- `SEAM_DP_INNER_POOL_CAP` (optional, default `1`): when `1`, each task’s inner thread count is also clamped to `fld(JULIA_NUM_THREADS, concurrent_outer_tasks)` where `concurrent_outer_tasks ≈ min(ntx, SEAM_WAVE_SIZE)` per batch. This limits **nested** `Threads.@threads` (many Dagger row chunks × inner split) from oversubscribing Julia’s **single** thread pool — a common cause of **descaling** past modest core counts. Set to `0` only for experiments.
- `SEAM_DP_INNER_MIN_SPAN` (optional, default `256`): minimum column span per chunk before inner threading splits it (avoids tiny parallel regions).
- `SEAM_DP_INNER_MIN_CHUNK` (optional, default `512`): minimum x-elements of work per inner thread. Effective inner threads are additionally capped by `fld(span, SEAM_DP_INNER_MIN_CHUNK)` to avoid expensive nested regions on narrow chunks.

### Strong scaling / HPC runtime

On large nodes, nested threading plus Dagger can hurt past modest core counts. Before comparing thread sweeps:

- Export `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` so underlying BLAS/OpenMP does not add extra threads.
- On multi-socket hosts, try binding the process to one NUMA node (e.g. `numactl --cpunodebind=0 --membind=0`; exact flags depend on your system) and compare to default.
- Avoid interrupting Julia mid-run during heavy Dagger work (`^C` can produce noisy shutdown traces).

Quick scaling probe (strong only, adjust `rows`/`tile_*` as needed):

```bash
for t in 1 16 32 64 100; do
  echo "threads=$t"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  SEAM_GPU=0 SEAM_DEVICE=cpu \
  julia --project=benchmarks/scripts -t"$t" -e '
    using DaggerAppsBenchmarks
    run_seam_carving(runs=3, rows=8192, cols=8192, k=1, tile_h=32, tile_w=1024,
      variants=[:cpu_dagger_wavefront, :cpu_dagger_triangles],
      device=:cpu, want_gpu=false, scenarios=:strong)
  '
done
```

**Baseline comparison (high `ntx` vs hybrid):** (1) small `tile_w` so `ntx` is large (many Dagger tasks, no inner threads; leave `SEAM_DP_INNER_THREADS` unset or `1`). (2) wider `tile_w` with `SEAM_DP_INNER_THREADS` set so `ntx × inner` is near `$t`. Check parity against `seam_carve_cpu_serial` on the same image.

**If time gets worse as `-t` grows (descaling):** Julia shares **one** thread pool. Many **concurrent** Dagger DP chunks (`ntx`), each entering `Threads.@threads` for the inner `x` split, creates **nested** parallel regions and often hurts. Prefer **fewer column chunks** (larger `SEAM_TILE_W` / `tile_w`, e.g. **1024 or 2048** for 8192-wide images → `ntx` 8 or 4) and **raise** `SEAM_DP_INNER_THREADS` so `ntx × inner ≈` thread count, with **`SEAM_DP_INNER_POOL_CAP=1`** (default) on. If chunks are still narrow, increase **`SEAM_DP_INNER_MIN_CHUNK`** (e.g. 1024) so each inner thread gets enough work.

### Westrick variant: strong-scaling sweep (52+ threads)

From `DaggerApps/`, use `benchmarks/scripts/seam-westrick-scaling.jl` to launch **one subprocess per thread count** (each with `julia -tN`), default sweep `1,2,4,8,16,32,52` (override with `SEAM_THREAD_SWEEP`). The driver sets `SEAM_VARIANTS` to `cpu_dagger_westrick` and `SEAM_GPU=0` unless you override them.

Example (Westrick-shaped image ~600×2600, [blog benchmark](https://shwestrick.github.io/2020/07/29/seam-carve.html)):

```bash
cd DaggerApps
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SEAM_THREAD_SWEEP=1,2,4,8,16,32,52 BENCH_RUNS=3 \\
  SEAM_ROWS=600 SEAM_COLS=2600 SEAM_TILE_H=300 SEAM_TILE_W=650 \\
  julia benchmarks/scripts/seam-westrick-scaling.jl
```

**Scaling expectations:** the same post reports only ~**9×** speedup at 30 cores for triangular DP versus ~**2×** for row-major, with memory bandwidth and barrier span as limits—not linear speedup in core count. The westrick variant targets **coarse Dagger tasks** and **two barriers per strip** to avoid the per-row barrier path in `seam_carve_cpu_dagger_triangles`.

### Tuning `cpu_dagger_westrick` toward 1–52 threads (Dagger)

Use these in order; combine **large enough** `SEAM_ROWS×SEAM_COLS` with **correct NUMA** (`numactl` CPUs + `membind` on one node).

1. **Job CPUs:** `nproc` (or PBS `ncpus`) must be **≥** max `julia -t`. Mixed NUMA + `--membind=0` while threads run on node 1 kills scaling.
2. **Problem size:** enlarge the image so DP + energy dominate fixed overhead (e.g. **2400×10400** or bigger if memory allows).
3. **Tiles (`SEAM_TILE_H` / `SEAM_TILE_W`):** aim for **many** energy/remove tasks but not microscopic tiles—e.g. `nty×ntx` roughly **2–8×** thread count. Example: **400×400** on 2400×10400 → **6×26=156** tiles.
4. **`SEAM_WESTRICK_BLOCK_WIDTH` (even):** default **80**. **Smaller** (e.g. **64**, **48**) → more triangles per phase → more Dagger tasks (more parallelism, more overhead). **Larger** (e.g. **96**, **112**) → fewer, heavier tasks—sometimes better past **~16** threads if overhead dominated.
5. **`SEAM_WAVE_SIZE`:** leave **unset** for full-phase waves (see above). Set explicitly only to **cap** concurrent spawns (debugging / memory).
6. **Weak scaling:** to see throughput grow with threads, run **`SEAM_SCENARIOS=weak`** and `SEAM_WEAK_SCALE=linear` or `sqrt` so work per thread stays roughly constant.
7. **`SEAM_K`:** `k>1` repeats the pipeline; strong scaling improves when **per-iteration** work is large relative to spawn/sync cost.
