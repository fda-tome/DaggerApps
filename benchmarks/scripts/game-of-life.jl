using BenchmarkTools
using Dates
using Dagger
using Printf
using Random
using Statistics

const APP = "game-of-life"
const APP_DIR = abspath(joinpath(@__DIR__, "..", "..", "apps", APP))
const APP_IMPL = joinpath(APP_DIR, "src", "DaggerGameOfLife.jl")
const RESULTS_APP_DIR = abspath(joinpath(@__DIR__, "..", "results", APP))

include(APP_IMPL)
if !isdefined(@__MODULE__, :DaggerGameOfLife)
    error("DaggerGameOfLife module not found after include($APP_IMPL).")
end
const DaggerGameOfLife = getfield(@__MODULE__, :DaggerGameOfLife)

const DEFAULT_VARIANTS = [
    :cpu_serial,
    :cpu_dagger_stencil_wrap,
    :cpu_dagger_stencil_pad,
]

function _parse_variants()
    raw = strip(get(ENV, "LIFE_VARIANTS", ""))
    if isempty(raw) || lowercase(raw) == "all"
        return DEFAULT_VARIANTS
    end
    tokens = split(raw, r"[,\s]+", keepempty=false)
    variants = Symbol[]
    for token in tokens
        push!(variants, Symbol(lowercase(token)))
    end
    unknown = setdiff(variants, DEFAULT_VARIANTS)
    if !isempty(unknown)
        error("Unknown LIFE_VARIANTS: $(join(string.(unknown), ", ")). Known: $(join(string.(DEFAULT_VARIANTS), ", "))")
    end
    return variants
end

function _thread_count()::Int
    return Threads.nthreads()
end

@inline _invoke0(f) = f()

function _bench_times(f, runs::Int)::Vector{Float64}
    f() # warmup (compile)
    trial = BenchmarkTools.@benchmark _invoke0($f) samples=runs evals=1
    return trial.times ./ 1e9
end

function _run_variant(
    variant::Symbol,
    initial::AbstractMatrix{Bool},
    steps::Int,
    block_h::Int,
    block_w::Int,
)
    if variant === :cpu_serial
        return DaggerGameOfLife.game_of_life_cpu_serial(initial; steps=steps, boundary=:wrap)
    elseif variant === :cpu_dagger_stencil_wrap
        return DaggerGameOfLife.game_of_life_dagger_stencil(initial;
            steps=steps,
            block_h=block_h,
            block_w=block_w,
            boundary=:wrap,
        )
    elseif variant === :cpu_dagger_stencil_pad
        return DaggerGameOfLife.game_of_life_dagger_stencil(initial;
            steps=steps,
            block_h=block_h,
            block_w=block_w,
            boundary=:pad,
            pad_value=false,
        )
    else
        error("Unknown game-of-life variant: $variant")
    end
end

function life_job(
    variant::Symbol,
    rows::Int,
    cols::Int,
    density::Float64,
    seed::Int,
    steps::Int,
    block_h::Int,
    block_w::Int,
)
    Random.seed!(seed)
    initial = rand(rows, cols) .< density
    out = _run_variant(variant, initial, steps, block_h, block_w)
    return DaggerGameOfLife.alive_count(out)
end

function _write_runs_csv(path::AbstractString, scenario::AbstractString, variant::Symbol, threads::Int, rows::Int, cols::Int,
                         steps::Int, block_h::Int, block_w::Int, density::Float64, times::Vector{Float64}; write_header::Bool=false)
    open(path, write_header ? "w" : "a") do io
        if write_header
            println(io, "scenario,variant,threads,rows,cols,steps,block_h,block_w,density,run,time_sec")
        end
        for (i, t) in enumerate(times)
            println(io, "$(scenario),$(variant),$(threads),$(rows),$(cols),$(steps),$(block_h),$(block_w),$(@sprintf("%.4f", density)),$(i),$(@sprintf("%.9f", t))")
        end
    end
end

function _bench_variant(variant::Symbol, rows::Int, cols::Int, density::Float64, steps::Int, block_h::Int, block_w::Int, runs::Int)
    seed = Ref(0)
    f = () -> begin
        seed[] += 1
        life_job(variant, rows, cols, density, seed[], steps, block_h, block_w)
    end
    return _bench_times(f, runs)
end

function _weak_dims(
    rows::Int,
    cols::Int,
    threads::Int;
    weak_scale=nothing,
    weak_rows::Union{Nothing, Int}=nothing,
    weak_cols::Union{Nothing, Int}=nothing,
)
    if weak_rows !== nothing || weak_cols !== nothing
        return weak_rows === nothing ? rows : weak_rows,
               weak_cols === nothing ? cols : weak_cols,
               nothing
    end

    if haskey(ENV, "LIFE_WEAK_ROWS") || haskey(ENV, "LIFE_WEAK_COLS")
        wrows = parse(Int, get(ENV, "LIFE_WEAK_ROWS", string(rows)))
        wcols = parse(Int, get(ENV, "LIFE_WEAK_COLS", string(cols)))
        return wrows, wcols, nothing
    end

    if weak_scale === nothing
        weak_scale = lowercase(strip(get(ENV, "LIFE_WEAK_SCALE", "sqrt")))
    end

    scale =
        weak_scale isa Symbol && weak_scale === :sqrt ? sqrt(threads) :
        weak_scale isa AbstractString && lowercase(strip(weak_scale)) == "sqrt" ? sqrt(threads) :
        weak_scale isa Symbol && weak_scale === :linear ? threads :
        weak_scale isa AbstractString && lowercase(strip(weak_scale)) == "linear" ? threads :
        weak_scale isa Real ? float(weak_scale) :
        parse(Float64, String(weak_scale))

    wrows = max(1, round(Int, rows * scale))
    wcols = max(1, round(Int, cols * scale))
    return wrows, wcols, scale
end

function _parse_scenarios()
    raw = lowercase(strip(get(ENV, "LIFE_SCENARIOS", "both")))
    if raw in ("both", "all", "")
        return ("strong", "weak")
    elseif raw in ("strong",)
        return ("strong",)
    elseif raw in ("weak",)
        return ("weak",)
    else
        error("Unknown LIFE_SCENARIOS=$raw. Use strong|weak|both.")
    end
end

function _normalize_scenarios(scenarios)
    scenarios === nothing && return _parse_scenarios()

    _one(x) = begin
        s = x isa Symbol ? lowercase(String(x)) : lowercase(strip(String(x)))
        s in ("both", "all") && return ("strong", "weak")
        s in ("strong", "weak") && return (s,)
        error("Unknown scenario=$x. Use :strong, :weak, or :both.")
    end

    if scenarios isa Symbol || scenarios isa AbstractString
        return _one(scenarios)
    elseif scenarios isa Tuple || scenarios isa AbstractVector
        out = String[]
        for x in scenarios
            for y in _one(x)
                y in out || push!(out, y)
            end
        end
        isempty(out) && error("Empty scenarios list.")
        return tuple(out...)
    else
        error("Invalid scenarios type: $(typeof(scenarios)). Use Symbol/String or a tuple/vector of them.")
    end
end

"""
    run_benchmark(; runs=3, rows=1024, cols=1024, steps=100, block_h=128, block_w=128,
                  density=0.25, variants=_parse_variants())

Runs strong- and weak-scaling benchmarks for Game of Life variants.

Configuration (environment variables):
- `BENCH_RUNS` (default: 3)
- `LIFE_ROWS` (default: 1024)
- `LIFE_COLS` (default: 1024)
- `LIFE_STEPS` (default: 100)
- `LIFE_BLOCK_H` (default: 128)
- `LIFE_BLOCK_W` (default: 128)
- `LIFE_DENSITY` (default: 0.25)
- `LIFE_VARIANTS` (default: all; comma/space-separated list)
- `LIFE_WEAK_SCALE` (default: sqrt; options: sqrt|linear|<float>)
- `LIFE_WEAK_ROWS` / `LIFE_WEAK_COLS` (override weak dimensions)
- `LIFE_SCENARIOS` (default: both; strong|weak|both)
"""
function run_benchmark(;
    runs::Int=parse(Int, get(ENV, "BENCH_RUNS", "3")),
    rows::Int=parse(Int, get(ENV, "LIFE_ROWS", "1024")),
    cols::Int=parse(Int, get(ENV, "LIFE_COLS", "1024")),
    steps::Int=parse(Int, get(ENV, "LIFE_STEPS", "100")),
    block_h::Int=parse(Int, get(ENV, "LIFE_BLOCK_H", "128")),
    block_w::Int=parse(Int, get(ENV, "LIFE_BLOCK_W", "128")),
    density::Float64=parse(Float64, get(ENV, "LIFE_DENSITY", "0.25")),
    variants::Vector{Symbol}=_parse_variants(),
    scenarios=nothing,
    weak_scale=nothing,
    weak_rows::Union{Nothing, Int}=nothing,
    weak_cols::Union{Nothing, Int}=nothing,
)
    rows > 0 || throw(ArgumentError("rows must be > 0"))
    cols > 0 || throw(ArgumentError("cols must be > 0"))
    steps >= 0 || throw(ArgumentError("steps must be >= 0"))
    block_h > 0 || throw(ArgumentError("block_h must be > 0"))
    block_w > 0 || throw(ArgumentError("block_w must be > 0"))
    0.0 <= density <= 1.0 || throw(ArgumentError("density must be in [0, 1]"))

    threads = _thread_count()
    weak_rows, weak_cols, weak_scale_used = _weak_dims(rows, cols, threads; weak_scale=weak_scale, weak_rows=weak_rows, weak_cols=weak_cols)
    scenarios = _normalize_scenarios(scenarios)

    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    out_dir = joinpath(RESULTS_APP_DIR, ts)
    mkpath(out_dir)

    println("="^70)
    println("GAME-OF-LIFE BENCHMARK (DaggerGameOfLife variants)")
    println("="^70)
    println("Threads: $threads")
    println("Runs: $runs")
    println("Strong size: $(rows)x$(cols)")
    if weak_scale_used === nothing
        println("Weak size: $(weak_rows)x$(weak_cols)")
    else
        println("Weak size: $(weak_rows)x$(weak_cols) (scale=$(round(weak_scale_used, digits=3)))")
    end
    println("Steps: $steps")
    println("Blocks: $(block_h)x$(block_w)")
    println("Density: $density")
    println("Variants: $(join(string.(variants), ", "))")
    println()

    for scenario in scenarios
        s_rows, s_cols, label = scenario == "strong" ? (rows, cols, "Strong") : (weak_rows, weak_cols, "Weak")
        println(">>> $label scaling (rows=$(s_rows), cols=$(s_cols))")
        csv_path = joinpath(out_dir, "$(scenario)_scaling.csv")
        first = true
        for variant in variants
            times = _bench_variant(variant, s_rows, s_cols, density, steps, block_h, block_w, runs)
            println(@sprintf("  %-28s mean=%.4fs  std=%.4fs", string(variant), mean(times), std(times; corrected=false)))
            _write_runs_csv(csv_path, scenario, variant, threads, s_rows, s_cols, steps, block_h, block_w, density, times; write_header=first)
            first = false
        end
        println()
    end

    println("Results written to: $out_dir")
    return out_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
