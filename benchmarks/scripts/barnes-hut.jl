using Distributed
using Dates
using Printf
using Statistics
using BenchmarkTools

"""
Add `nw` Julia workers.  When running inside a PBS job (`\$PBS_NODEFILE` exists),
workers are distributed across allocated compute nodes via SSH (one per node).
Otherwise, workers are added locally (interactive / single-node use).
"""
function _add_workers(nw::Int)
    nodefile = get(ENV, "PBS_NODEFILE", "")
    if !isempty(nodefile) && isfile(nodefile)
        hosts = unique(readlines(nodefile))
        n = min(nw, length(hosts))
        julia_exec = joinpath(Sys.BINDIR, "julia")
        println("Distributing $n worker(s) across PBS nodes: ", hosts[1:n])
        flush(stdout)
        addprocs(
            hosts[1:n];
            exename=julia_exec,
            exeflags="--project=$(Base.active_project())",
            topology=:master_worker,
        )
    else
        addprocs(nw)
    end
end

# Add workers before loading Dagger (required by Dagger.jl). Set BARNES_NPROCS for a full multi-worker run.
if haskey(ENV, "BARNES_NPROCS") && nprocs() == 1
    nw = parse(Int, ENV["BARNES_NPROCS"])
    nw > 0 || error("BARNES_NPROCS must be positive, got: $(ENV["BARNES_NPROCS"])")
    _add_workers(nw)
end
using Dagger

const APP = "barnes-hut"
const APP_DIR = abspath(joinpath(@__DIR__, "..", "..", "apps", APP))
const APP_IMPL = joinpath(APP_DIR, "src", "DaggerBarnesHut.jl")
const RESULTS_APP_DIR = abspath(joinpath(@__DIR__, "..", "results", APP))

@everywhere include($APP_IMPL)
if !isdefined(@__MODULE__, :DaggerBarnesHut)
    error("DaggerBarnesHut module not found after include($APP_IMPL).")
end
const DaggerBarnesHut = getfield(@__MODULE__, :DaggerBarnesHut)
const bmark = DaggerBarnesHut.bmark

function _dagger_processors()::Int
    return length(Dagger.compatible_processors())
end

function _time_sec_bt(f; bt_samples::Int=1, bt_evals::Int=1)::Float64
    bt_samples >= 1 || throw(ArgumentError("BENCH_BT_SAMPLES must be >= 1"))
    bt_evals >= 1 || throw(ArgumentError("BENCH_BT_EVALS must be >= 1"))
    # Use BenchmarkTools and return the median over samples for stability.
    trial = @benchmark $f() samples=bt_samples evals=bt_evals
    return BenchmarkTools.median(trial).time / 1e9
end

function _run_n(f, n::Int; bt_samples::Int=1, bt_evals::Int=1)::Vector{Float64}
    times = Vector{Float64}(undef, n)
    for i in 1:n
        GC.gc()
        times[i] = _time_sec_bt(f; bt_samples=bt_samples, bt_evals=bt_evals)
    end
    return times
end

function _write_runs_csv(path::AbstractString, scenario::AbstractString, procs::Int, N::Int, theta::Float64, times::Vector{Float64})
    open(path, "w") do io
        println(io, "scenario,dagger_processors,N,theta,run,time_sec")
        for (i, t) in enumerate(times)
            println(io, scenario, ",", procs, ",", N, ",", theta, ",", i, ",", Printf.@sprintf("%.9f", t))
        end
    end
end

"""
    run_benchmark(; runs=3, theta=0.5)

Runs both a strong-scaling and weak-scaling measurement for the current Dagger processor configuration.

Configuration (environment variables):
- `BARNES_NPROCS` – number of worker processes to add before running (e.g. 32 for a 32-worker run). Must be set before the script loads; workers are added before loading Dagger.
- `BENCH_RUNS` (default: 3)
- Timings are collected with `BenchmarkTools` (`@benchmark`), recording median time per run.
- `BENCH_BT_SAMPLES` (default: 5) — BenchmarkTools samples per recorded run
- `BENCH_BT_EVALS` (default: 1) — BenchmarkTools evals per sample
- `BENCH_WARMUP` (default: 1) — untimed `bmark` runs before strong and before weak (JIT / precompile); set `0` to disable
- `BARNES_THETA` (default: 0.5)
- `BARNES_N_STRONG` (default: 250000) — fixed N for strong scaling
- `BARNES_BODIES_PER_PROC` (default: 20000) — weak scaling uses N = this × dagger_processors
"""
function run_benchmark(;
    runs::Int=parse(Int, get(ENV, "BENCH_RUNS", "3")),
    warmup::Int=parse(Int, get(ENV, "BENCH_WARMUP", "1")),
    theta::Float64=parse(Float64, get(ENV, "BARNES_THETA", "0.5")),
    bt_samples::Int=parse(Int, get(ENV, "BENCH_BT_SAMPLES", "5")),
    bt_evals::Int=parse(Int, get(ENV, "BENCH_BT_EVALS", "1")),
)
    procs = _dagger_processors()

    strong_N = parse(Int, get(ENV, "BARNES_N_STRONG", "250000"))
    weak_bodies_per_proc = parse(Int, get(ENV, "BARNES_BODIES_PER_PROC", "20000"))
    weak_N = max(1, weak_bodies_per_proc * max(1, procs))

    out_dir = get(ENV, "BARNES_RESULTS_DIR", "") |> d -> isempty(d) ? joinpath(RESULTS_APP_DIR, Dates.format(Dates.now(), "yyyymmdd_HHMMSS")) : abspath(d)
    mkpath(out_dir)

    println("="^70)
    println("BARNES-HUT BENCHMARK")
    println("="^70)
    println("Dagger processors: $procs")
    println("Runs: $runs")
    println("Warmup runs (untimed): $warmup")
    println("Theta: $theta")
    println("BenchmarkTools samples/evals per run: $bt_samples/$bt_evals")
    println()
    flush(stdout)

    if warmup > 0
        println(">>> Warmup — strong (N=$strong_N), $warmup untimed run(s) (JIT / precompile)")
        for _ in 1:warmup
            bmark(strong_N, theta)
        end
        GC.gc()
        println()
    end

    println(">>> Strong scaling (fixed N=$strong_N)")
    strong_times = _run_n(() -> bmark(strong_N, theta), runs; bt_samples=bt_samples, bt_evals=bt_evals)
    println(Printf.@sprintf("  mean=%.4fs  std=%.4fs", mean(strong_times), std(strong_times)))
    flush(stdout)

    if warmup > 0
        println()
        println(">>> Warmup — weak (N=$weak_N), $warmup untimed run(s)")
        for _ in 1:warmup
            bmark(weak_N, theta)
        end
        GC.gc()
        println()
    end

    println(">>> Weak scaling (N = bodies_per_proc * procs = $weak_bodies_per_proc * $procs = $weak_N)")
    weak_times = _run_n(() -> bmark(weak_N, theta), runs; bt_samples=bt_samples, bt_evals=bt_evals)
    println(Printf.@sprintf("  mean=%.4fs  std=%.4fs", mean(weak_times), std(weak_times)))
    flush(stdout)

    _write_runs_csv(joinpath(out_dir, "strong_scaling.csv"), "strong", procs, strong_N, theta, strong_times)
    _write_runs_csv(joinpath(out_dir, "weak_scaling.csv"), "weak", procs, weak_N, theta, weak_times)

    println()
    println("Results written to: $out_dir")
    flush(stdout)
    return out_dir
end

"""
    run_scaling_experiment(; worker_counts=[1,2,4,8,16,32], kwargs...)

Runs strong and weak scaling by launching a separate Julia process for each worker count.
Results are written to `benchmarks/results/barnes-hut/scaling_<timestamp>/` with merged
`scaling_strong.csv` and `scaling_weak.csv` (each row includes `dagger_processors`).

Environment variables passed through to each run: `BENCH_RUNS`, `BENCH_WARMUP`, `BENCH_BT_SAMPLES`, `BENCH_BT_EVALS`, `BARNES_THETA`,
`BARNES_N_STRONG`, `BARNES_BODIES_PER_PROC`.
"""
function run_scaling_experiment(;
    worker_counts::Vector{Int}=[1, 2, 4, 8, 16, 32],
)
    repo_root = abspath(joinpath(@__DIR__, "..", ".."))
    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    out_base = joinpath(RESULTS_APP_DIR, "scaling_" * ts)
    mkpath(out_base)

    println("="^70)
    println("BARNES-HUT STRONG / WEAK SCALING EXPERIMENT (up to $(maximum(worker_counts)) workers)")
    println("="^70)
    println("Worker counts: ", worker_counts)
    println("Output base: ", out_base)
    println()
    flush(stdout)

    for w in worker_counts
        w_dir = joinpath(out_base, string(w))
        mkpath(w_dir)
        env = copy(ENV)
        env["BARNES_NPROCS"] = string(w)
        env["BARNES_RESULTS_DIR"] = w_dir
        delete!(env, "BARNES_SCALING_EXPERIMENT")  # so child runs run_benchmark(), not this again
        println(">>> Running with $w worker(s) ...")
        flush(stdout)
        julia_cmd = Base.julia_cmd()
        bench_script = joinpath(repo_root, "benchmarks", "scripts", "barnes-hut.jl")
        proj = joinpath(repo_root, "apps", APP)
        cmd = setenv(`$julia_cmd --project=$proj $bench_script`, env)
        run(cmd)
        println()
        flush(stdout)
    end

    # Merge CSVs into scaling_strong.csv and scaling_weak.csv
    strong_lines = String[]
    weak_lines = String[]
    header_strong = "scenario,dagger_processors,N,theta,run,time_sec"
    header_weak = header_strong
    for w in worker_counts
        w_dir = joinpath(out_base, string(w))
        for (name, lines) in [("strong_scaling.csv", strong_lines), ("weak_scaling.csv", weak_lines)]
            path = joinpath(w_dir, name)
            isfile(path) || continue
            content = read(path, String)
            raw_lines = split(content, '\n'; keepempty=false)
            isempty(raw_lines) && continue
            for (i, ln) in enumerate(raw_lines)
                i == 1 && startswith(ln, "scenario") && continue  # skip header
                push!(lines, ln)
            end
        end
    end
    open(joinpath(out_base, "scaling_strong.csv"), "w") do io
        println(io, header_strong)
        for ln in strong_lines
            println(io, ln)
        end
    end
    open(joinpath(out_base, "scaling_weak.csv"), "w") do io
        println(io, header_weak)
        for ln in weak_lines
            println(io, ln)
        end
    end

    println("Merged results: $(out_base)/scaling_strong.csv, scaling_weak.csv")
    flush(stdout)
    return out_base
end

function _shutdown_dagger_and_workers()
    try
        state = Dagger.Sch.EAGER_STATE[]
        if state !== nothing && !state.halt.set
            Dagger.cancel!(; halt_sch=true)
        end
    catch end
    try
        wkrs = workers()
        length(wkrs) > 0 && rmprocs(wkrs; waitfor=30.0)
    catch end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if get(ENV, "BARNES_SCALING_EXPERIMENT", "") == "1"
        run_scaling_experiment()
    else
        try
            run_benchmark()
        finally
            _shutdown_dagger_and_workers()
        end
    end
end
