# Strong-scaling helper for `cpu_dagger_westrick` (Westrick strip DP + Dagger).
#
# Usage (from `DaggerApps/`):
#   SEAM_THREAD_SWEEP=1,2,4,8,16,32,52 \\
#   BENCH_RUNS=1 SEAM_ROWS=600 SEAM_COLS=2600 SEAM_TILE_H=300 SEAM_TILE_W=650 \\
#   julia benchmarks/scripts/seam-westrick-scaling.jl
#
# Each subprocess is launched with `julia -tN` for N in the sweep list. Set `OMP_NUM_THREADS=1`
# and BLAS thread env vars as in `apps/seam-carving/README.md` before running.
#
# Optional NUMA prefix on every subprocess (use on bound compute nodes):
#   export SEAM_NUMA_CPUBIND=+0-+51
#   export SEAM_NUMA_MEMBIND=0   # default when SEAM_NUMA_CPUBIND is set
# Other exports (`SEAM_ROWS`, `SEAM_PIN_THREADS`, `SEAM_WESTRICK_PHASE_TASKS_GE_THREADS`, …) are inherited.

const SCRIPT_DIR = abspath(@__DIR__)
const DAGGER_APPS = abspath(joinpath(SCRIPT_DIR, "..", ".."))
const SEAM_PROJECT = joinpath(DAGGER_APPS, "apps", "seam-carving")
const SEAM_BENCH = joinpath(SCRIPT_DIR, "seam-carving.jl")

function _parse_thread_sweep()::Vector{Int}
    raw = strip(get(ENV, "SEAM_THREAD_SWEEP", "1,2,4,8,16,32,52"))
    isempty(raw) && return [1]
    return parse.(Int, split(raw, ','; keepempty=false))
end

function main()
    jcmd = Base.julia_cmd()
    bind = strip(get(ENV, "SEAM_NUMA_CPUBIND", ""))
    mb = strip(get(ENV, "SEAM_NUMA_MEMBIND", "0"))
    for t in _parse_thread_sweep()
        # Plain `julia ...` first, optional `numactl ... julia ...`, then `addenv` on the **outer**
        # command only. Nesting `addenv(cmd)` inside `` `numactl $cmd` `` throws:
        # `ArgumentError: Non-default environment behavior is only permitted for the first interpolant.`
        runner = if isempty(bind)
            `$jcmd -t$t --project=$SEAM_PROJECT $SEAM_BENCH`
        else
            `numactl --physcpubind=$bind --membind=$mb $jcmd -t$t --project=$SEAM_PROJECT $SEAM_BENCH`
        end
        cmd = addenv(
            runner,
            "SEAM_VARIANTS" => get(ENV, "SEAM_VARIANTS", "cpu_dagger_westrick"),
            "SEAM_GPU" => get(ENV, "SEAM_GPU", "0"),
            "SEAM_SCENARIOS" => get(ENV, "SEAM_SCENARIOS", "strong"),
        )
        println(stderr, "=== Westrick scaling: subprocess -t$t ===")
        run(cmd)
    end
end

main()
