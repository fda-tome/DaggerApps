# Dagger GPU Cholesky benchmark app

Blocked `LinearAlgebra.cholesky` on a GPU-backed `Dagger.DArray` with **four devices** and a **2×2 block-cyclic** `assignment` matrix of Dagger `Processor`s (ScaLAPACK-style 2D decomposition). The SPD matrix is **`ones`** plus a diagonal boost applied only on diagonal tiles (no host `collect` for setup).

## Requirements

- **Julia** ≥ 1.10
- **Four GPUs** of one vendor (CUDA, AMDGPU, oneAPI, or Metal)
- Load the GPU package **before** `Dagger` in the same session ([Dagger.jl docs](https://juliaparallel.org/Dagger.jl/dev/))

## Dagger.jl from `master`

The environment pins **Dagger** to [JuliaParallel/Dagger.jl](https://github.com/JuliaParallel/Dagger.jl) `master` via `Project.toml` `[sources]` and `Manifest.toml`. To refresh to the latest commit:

```julia
using Pkg
Pkg.update("Dagger")
```

Or reinstall:

```julia
Pkg.add(Pkg.PackageSpec(url="https://github.com/JuliaParallel/Dagger.jl", rev="master"))
```

If `Pkg.instantiate()` reports a tree hash mismatch, run `Pkg.resolve()` or update the git entry as above.

## Run the benchmark

From the `DaggerApps` repo root (adjust paths if needed):

```bash
julia --project=apps/gpu-cholesky -e '
    using CUDA   # or AMDGPU, oneAPI, Metal
    using Dagger
    include("benchmarks/scripts/gpu-cholesky.jl")
    run_benchmark()
'
```

Default sweep: `N = 2^k` for `k = 10:18` (up to **262144×262144**). Override with env vars (see script header in `benchmarks/scripts/gpu-cholesky.jl`).

**Vendor comparison:** the benchmark script also times **single-GPU** `LinearAlgebra.cholesky!` (vendor **potrf**: cuSOLVER, rocSOLVER, Intel oneMKL, or Metal as appropriate) on a dense GPU matrix with the **same** SPD values as the Dagger `DArray` (`ones` plus diagonal boost). That timing includes restoring the matrix with `copyto!` each trial so every run is a valid factorization. Set `CHOLESKY_VENDOR=0` to skip. Use `CHOLESKY_VENDOR_DEVICE` (default `0`, 0-based) to pick the device for the dense baseline. The Dagger path still uses **four** GPUs; the vendor path is a **one-GPU** reference.

**Dagger performance metrics:** set `CHOLESKY_PERF_LOG=1` to turn on Dagger’s TimespanLogging (`enable_logging!` with `metrics=true` and task function names). Each `(N, block_size)` appends one **NDJSON** line to `perf_dagger.jsonl` under the same timestamped results folder as `cholesky_times.csv` (override with `CHOLESKY_PERF_LOG_PATH`). Use `CHOLESKY_PERF_SCOPE=timed` (default) to record logs only during the timed Dagger trials (after warmup), or `full` to include matrix construction and warmup as well. Summaries aggregate paired `:core` span durations by category (e.g. `:compute`, `:move`) and list frequent task function names. Logging adds overhead; use for diagnosis, not bare-metal throughput runs.

**Tuning tile size:** Cholesky on a `DArray` depends strongly on **`CHOLESKY_BLOCK`** (must divide `N`). Set **`CHOLESKY_BLOCKS=256,512,1024`** (comma-separated) to sweep several tile sizes in one run; the CSV gets one row per `(N, block_size)`, while the **vendor** dense baseline is still timed **once per `N`** (it does not depend on the Dagger tile). Try powers of two near a kernel-friendly size for your GPU.

**Algorithm variants:** **`CHOLESKY_ALGO=ll`** (default) selects the Cholesky algorithm for the SC26 paper / AD (left-looking blocked; matches the paper narrative). Set a comma-separated list to sweep variants in a single run:

| Value   | Description |
|---------|-------------|
| `rl`    | Right-looking (original Dagger, no processor pinning or lookahead). Useful as a baseline. |
| `rl_la` | Right-looking + **processor pinning** (each task runs on the GPU owning its output tile) + **lookahead** spawn ordering (critical-path `trsm`/`syrk` for column `k+1` submitted before bulk updates). |
| `ll`    | Left-looking + **processor pinning**. Concentrates GEMM-based updates on the current column before factorizing (per Haidar et al. 2017, Algorithm 1). **Default** for the artifact. |

Example sweep: `CHOLESKY_ALGO=rl,rl_la,ll`. The CSV and NDJSON perf log include an `algorithm` column/field.

**In-place Dagger path:** **`CHOLESKY_INPLACE=1`** times `LinearAlgebra.cholesky!` with a **`copyto!`** from a saved template each repetition (same pattern as the vendor benchmark). That can reduce allocation overhead from repeated out-of-place `cholesky` for some sizes, at the cost of **`~2×`** `DArray` GPU memory per row while the template exists.

**Memory:** at large `N`, dense data dominates; prefer **`Float32`** (`CHOLESKY_ELTYPE=Float32`, default) and lower `CHOLESKY_K_MAX` if you hit OOM. The vendor baseline keeps **two** full `N×N` tiles on one GPU (`template` + `workspace`), in addition to the distributed Dagger storage.

## ALCF Polaris compute nodes (interactive / PBS jobs)

- **`git clone git@github.com:...` often fails** on compute nodes (no outbound SSH). Clone or `Pkg.add` from a **login node**, or use **HTTPS** with the ALCF proxy. To use a local checkout instead of the GitHub pin: `Pkg.develop(path="...")` for **Dagger.jl**.
- **`rename ... cross-device link not permitted (EXDEV)`** during `Pkg` registry updates: PBS may use `/var/tmp` for downloads while `~/.julia` is on another filesystem. Point temp space at the **same filesystem** as your depot, for example:
  ```bash
  export TMPDIR="${HOME}/.julia/tmp"
  mkdir -p "$TMPDIR"
  ```
  Or put the whole depot on Eagle if home quota is tight:
  ```bash
  export JULIA_DEPOT_PATH="/lus/eagle/projects/<project>/<user>/.julia"
  mkdir -p "$JULIA_DEPOT_PATH"
  ```
  (Use a path under your Eagle allocation; avoid cross-filesystem renames from job `TMPDIR`.)

## AMD MI300 / ROCm batch jobs (SLURM)

- Load your site’s **ROCm** module (and Julia) on the **login or batch node** as required.
- This benchmark needs **four visible GPU devices** on one node (`HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` if you must pin devices). See `scripts/amd_rocm_env.sh` for optional exports.
- Example batch driver (same sweep as `scripts/polaris_cholesky_bench.pbs`, but `using AMDGPU` instead of `using CUDA`):

  ```bash
  export PROJ=/path/to/parent/of/DaggerApps
  sbatch scripts/mi300a_cholesky_bench.slurm
  ```

  Set `JULIA`, `JULIA_PROJECT`, `LOGDIR`, and `#SBATCH` account/partition in that script for your facility.

- One-line interactive test from `DaggerApps` root:

  ```bash
  CHOLESKY_DEVICE=amdgpu CHOLESKY_NS=1024 CHOLESKY_BLOCK=512 \
    julia --project=apps/gpu-cholesky -e 'using AMDGPU; using Dagger; include("benchmarks/scripts/gpu-cholesky.jl"); run_benchmark()'
  ```

## Library API

`using DaggerGpuCholesky` (after `using` your GPU stack and `Dagger`) provides helpers such as `four_gpu_processors`, `cholesky_block_cyclic_assignment`, `spd_ones_darray`, `bench_cholesky_once!`, and `spd_ones_dense_vendor` / `bench_vendor_cholesky_once!` for the dense single-GPU vendor baseline (backend follows whichever GPU package is loaded in `Main`).
