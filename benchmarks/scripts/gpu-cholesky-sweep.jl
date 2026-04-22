# Full sweep: precompile once, then run all (N, grid, algo) combos with 1 trial, 0 warmup.
# Usage: julia --project=DaggerApps/apps/gpu-cholesky DaggerApps/benchmarks/scripts/gpu-cholesky-sweep.jl

using CUDA
using Dagger

include("DaggerApps/benchmarks/scripts/gpu-cholesky.jl")

const _DG = DaggerGpuCholesky

# ── GPU inventory ──
println("=== GPU inventory ===")
for d in CUDA.devices()
    CUDA.device!(d)
    free, total = CUDA.memory_info()
    println("  GPU $(CUDA.deviceid(d)): $(round(free/2^30; digits=2)) / $(round(total/2^30; digits=2)) GiB free")
end

# ── Precompile both algorithms on a tiny 128×128 DArray (2×2 grid) ──
# Uses the same processor selection as gpu-cholesky.jl (`CHOLESKY_NUM_GPUS`, default 4).
println("\n=== Precompiling (128×128, 2×2) ... ==="); flush(stdout)
_procs = _DG.gpu_processors_for_cholesky()
_asg = _DG.cholesky_tile_assignment(_procs, 2)
_DA = _DG.spd_ones_darray(Float32, 128, 64, _asg)
_precompile_algos = Main._effective_algo_list_for_gpu_count([:rl_la, :ll], length(_procs))
for algo in _precompile_algos
    Dagger.with(Dagger.CHOLESKY_ALGORITHM => algo) do
        Dagger.darray_cholesky!(copy(_DA))
    end
end
_DG.unsafe_free_darray!(_DA)
_DG.purge_gpu_memory!()
println("Precompilation done.\n"); flush(stdout)

# ── Sweep definition ──
ns     = [4096, 8192, 16384, 32768, 65536, 131072]
grids  = [2, 4, 8]   # n×n tile grids → block_size = N/n

# Common env
ENV["CHOLESKY_TRIALS"]   = "1"
ENV["CHOLESKY_WARMUP"]   = "0"
ENV["CHOLESKY_ALGO"]     = "rl_la,ll"
ENV["CHOLESKY_VENDOR"]   = "1"
ENV["CHOLESKY_INPLACE"]  = "1"
ENV["CHOLESKY_PERF_LOG"] = "0"
ENV["CHOLESKY_CHECK"]    = "0"

for N in ns
    block_sizes = [div(N, g) for g in grids]
    ENV["CHOLESKY_NS"]     = string(N)
    ENV["CHOLESKY_BLOCKS"] = join(block_sizes, ",")
    println("=== N=$N  blocks=$(ENV["CHOLESKY_BLOCKS"]) ==="); flush(stdout)

    for d in CUDA.devices()
        CUDA.device!(d)
        free, _ = CUDA.memory_info()
        println("  GPU $(CUDA.deviceid(d)): $(round(free/2^30; digits=2)) GiB free")
    end

    try
        run_benchmark()
    catch ex
        @error "FAILED N=$N" exception=(ex, catch_backtrace())
    end

    _DG.purge_gpu_memory!()
    println()
end

println("=== Sweep complete ===")
