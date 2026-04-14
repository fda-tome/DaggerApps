using BenchmarkTools
using Dates
using Dagger
using Printf
using Random
using Statistics

const APP = "seam-carving"
const APP_DIR = abspath(joinpath(@__DIR__, "..", "..", "apps", APP))
const APP_IMPL = joinpath(APP_DIR, "src", "DaggerSeamCarving.jl")
const RESULTS_APP_DIR = abspath(joinpath(@__DIR__, "..", "results", APP))

const DEVICE_BACKENDS = (
    (:cuda, :CUDA, :CuArray),
    (:amdgpu, :AMDGPU, :ROCArray),
    (:oneapi, :oneAPI, :oneArray),
    (:metal, :Metal, :MtlArray),
)

function _parse_device()
    raw = lowercase(strip(get(ENV, "SEAM_DEVICE", "auto")))
    if raw in ("", "auto")
        return :auto
    elseif raw in ("cpu", "none")
        return :cpu
    elseif raw in ("cuda", "nvidia")
        return :cuda
    elseif raw in ("amdgpu", "rocm", "amd")
        return :amdgpu
    elseif raw in ("oneapi", "intel")
        return :oneapi
    elseif raw in ("metal", "apple")
        return :metal
    else
        error("Unknown SEAM_DEVICE: $raw. Use auto|cpu|cuda|amdgpu|oneapi|metal.")
    end
end

function _device_from_loaded()
    for (device, modsym, ctor) in DEVICE_BACKENDS
        if isdefined(Main, modsym)
            mod = getfield(Main, modsym)
            if isdefined(mod, ctor)
                return device
            end
        end
    end
    return :cpu
end

function _resolve_device(device::Symbol)
    return device === :auto ? _device_from_loaded() : device
end

include(APP_IMPL)
if !isdefined(@__MODULE__, :DaggerSeamCarving)
    error("DaggerSeamCarving module not found after include($APP_IMPL).")
end
const DaggerSeamCarving = getfield(@__MODULE__, :DaggerSeamCarving)

DaggerSeamCarving.seam_configure_thread_pinning!()

function _device_convert(device::Symbol, img)
    device = device === :auto ? _device_from_loaded() : device
    if device === :cpu
        return img
    end
    for (key, modsym, ctor) in DEVICE_BACKENDS
        if key === device
            if !isdefined(Main, modsym)
                error("Backend module $modsym not loaded. Run `using $modsym` before include.")
            end
            mod = getfield(Main, modsym)
            if !isdefined(mod, ctor)
                error("Backend module $modsym does not define $ctor.")
            end
            return getfield(mod, ctor)(img)
        end
    end
    error("Unknown device: $device")
end

function _run_variant(variant::Symbol, img, k::Int, tile_h::Int, tile_w::Int)
    if variant === :cpu_serial
        return DaggerSeamCarving.seam_carve_cpu_serial(img; k=k)
    elseif variant === :cpu_dagger
        return DaggerSeamCarving.seam_carve_cpu_dagger(img; k=k)
    elseif variant === :cpu_dagger_tiled
        return DaggerSeamCarving.seam_carve_cpu_dagger_tiled(img; k=k, tile_h=tile_h, tile_w=tile_w)
    elseif variant === :cpu_dagger_wavefront
        return DaggerSeamCarving.seam_carve_cpu_dagger_wavefront(img; k=k, tile_h=tile_h, tile_w=tile_w)
    elseif variant === :cpu_dagger_tileoverlap
        return DaggerSeamCarving.seam_carve_cpu_dagger_tileoverlap(img; k=k, tile_h=tile_h, tile_w=tile_w)
    elseif variant === :cpu_dagger_triangles
        return DaggerSeamCarving.seam_carve_cpu_dagger_triangles(img; k=k, tile_h=tile_h, tile_w=tile_w)
    elseif variant === :cpu_dagger_westrick
        ens = let raw = lowercase(strip(get(ENV, "SEAM_WESTRICK_PHASE_TASKS_GE_THREADS", "0")))
            raw in ("1", "true", "yes", "on")
        end
        return DaggerSeamCarving.seam_carve_cpu_dagger_westrick(img; k=k, tile_h=tile_h, tile_w=tile_w,
            ensure_phase_tasks_ge_threads = ens)
    elseif variant === :gpu_dagger
        return DaggerSeamCarving.seam_carve_gpu_dagger(img; k=k)
    elseif variant === :gpu_dagger_device
        return DaggerSeamCarving.seam_carve_gpu_dagger_device(img; k=k)
    else
        error("Unknown seam-carving variant: $variant")
    end
end

function seam_job(variant::Symbol, rows::Int, cols::Int, seed::Int, k::Int, tile_h::Int, tile_w::Int, device::Symbol)
    Random.seed!(seed)
    img = rand(Float32, rows, cols)
    if variant === :gpu_dagger || variant === :gpu_dagger_device
        img = _device_convert(device, img)
    end
    return _run_variant(variant, img, k, tile_h, tile_w)
end

const DEFAULT_VARIANTS = [
    :cpu_serial,
    :cpu_dagger,
    :cpu_dagger_tiled,
    :cpu_dagger_wavefront,
    :cpu_dagger_tileoverlap,
    :cpu_dagger_triangles,
    :cpu_dagger_westrick,
    :gpu_dagger,
    :gpu_dagger_device,
]

const GPU_VARIANTS = Set([:gpu_dagger, :gpu_dagger_device])

function _parse_variants()
    raw = strip(get(ENV, "SEAM_VARIANTS", ""))
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
        error("Unknown SEAM_VARIANTS: $(join(string.(unknown), ", ")). Known: $(join(string.(DEFAULT_VARIANTS), ", "))")
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

function _write_runs_csv(path::AbstractString, scenario::AbstractString, variant::Symbol, threads::Int, rows::Int, cols::Int,
                         k::Int, tile_h::Int, tile_w::Int, times::Vector{Float64}; write_header::Bool=false)
    open(path, write_header ? "w" : "a") do io
        if write_header
            println(io, "scenario,variant,threads,rows,cols,k,tile_h,tile_w,run,time_sec")
        end
        for (i, t) in enumerate(times)
            println(io, "$(scenario),$(variant),$(threads),$(rows),$(cols),$(k),$(tile_h),$(tile_w),$(i),$(@sprintf("%.9f", t))")
        end
    end
end

function _bench_variant(variant::Symbol, rows::Int, cols::Int, k::Int, tile_h::Int, tile_w::Int, device::Symbol, runs::Int)
    seed = Ref(0)
    f = () -> begin
        seed[] += 1
        seam_job(variant, rows, cols, seed[], k, tile_h, tile_w, device)
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

    if haskey(ENV, "SEAM_WEAK_ROWS") || haskey(ENV, "SEAM_WEAK_COLS")
        wrows = parse(Int, get(ENV, "SEAM_WEAK_ROWS", string(rows)))
        wcols = parse(Int, get(ENV, "SEAM_WEAK_COLS", string(cols)))
        return wrows, wcols, nothing
    end

    if weak_scale === nothing
        weak_scale = lowercase(strip(get(ENV, "SEAM_WEAK_SCALE", "sqrt")))
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
    raw = lowercase(strip(get(ENV, "SEAM_SCENARIOS", "both")))
    if raw in ("both", "all", "")
        return ("strong", "weak")
    elseif raw in ("strong",)
        return ("strong",)
    elseif raw in ("weak",)
        return ("weak",)
    else
        error("Unknown SEAM_SCENARIOS=$raw. Use strong|weak|both.")
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
    run_benchmark(; runs=3, rows=512, cols=512, k=1, tile_h=150, tile_w=150, variants=_parse_variants(), device=_parse_device())

Runs strong- and weak-scaling benchmarks for the seam-carving variants using BenchmarkTools. Strong scaling uses the base
rows/cols; weak scaling scales rows/cols with the thread count (or an explicit override).

Configuration (environment variables):
- `BENCH_RUNS` (default: 3)
- `SEAM_ROWS` (default: 512)
- `SEAM_COLS` (default: 512)
- `SEAM_K` (default: 1)
- `SEAM_TILE_H` (default: 150)
- `SEAM_TILE_W` (default: 150)
- `SEAM_VARIANTS` (default: all; comma/space-separated list)
- `SEAM_GPU` (default: 1; set to 0 to skip GPU variants)
- `SEAM_DEVICE` (default: auto; cpu|cuda|amdgpu|oneapi|metal)
- `SEAM_WEAK_SCALE` (default: sqrt; options: sqrt|linear|<float>)
- `SEAM_WEAK_ROWS` / `SEAM_WEAK_COLS` (override weak dimensions)
- `SEAM_SCENARIOS` (default: both; strong|weak|both)

Wavefront/triangles tuning (see `apps/seam-carving/README.md`): `SEAM_WAVE_SIZE`, `SEAM_CPU_NESTED_THREADS`,
`SEAM_MIN_DP_TILE_W`, `SEAM_DP_INNER_THREADS`, `SEAM_DP_INNER_MIN_SPAN`. Westrick DP: `SEAM_WESTRICK_BLOCK_WIDTH`, `SEAM_WESTRICK_BLOCK_WIDTH_AUTO` / `SEAM_WESTRICK_BLOCK_WIDTH_AUTO_MAX`.
`SEAM_WESTRICK_PHASE_TASKS_GE_THREADS=1`: shrink tiling / block width so each Westrick Dagger phase has at least `Threads.nthreads()` tasks (`DaggerSeamCarving.westrick_ensure_tasks_per_thread`).
Strong-scaling sweeps: `benchmarks/scripts/seam-westrick-scaling.jl` and `SEAM_THREAD_SWEEP`.
NDJSON westrick perf: `SEAM_DEBUG_AGENT=1` or `SEAM_PERF_LOG=1` writes schedule + phase times + alloc/GC lines to `.cursor/debug-d35653.log` in the repo.

Keyword arguments:
- `scenarios`: override which scenarios to run (`:strong`, `:weak`, `:both` or a tuple/vector of them). When provided, this
  takes precedence over `SEAM_SCENARIOS`.
- `want_gpu`: override GPU variant filtering (when `false`, GPU variants are removed). When `nothing`, `SEAM_GPU` is used.
- `weak_scale`: override weak scaling rule (`:sqrt`, `:linear`, or a numeric factor). When `nothing`, `SEAM_WEAK_SCALE` is used.
- `weak_rows` / `weak_cols`: explicit weak dimensions override (takes precedence over `weak_scale` / `SEAM_WEAK_*`).
"""
function run_benchmark(;
    runs::Int=parse(Int, get(ENV, "BENCH_RUNS", "3")),
    rows::Int=parse(Int, get(ENV, "SEAM_ROWS", "512")),
    cols::Int=parse(Int, get(ENV, "SEAM_COLS", "512")),
    k::Int=parse(Int, get(ENV, "SEAM_K", "1")),
    tile_h::Int=parse(Int, get(ENV, "SEAM_TILE_H", "150")),
    tile_w::Int=parse(Int, get(ENV, "SEAM_TILE_W", "150")),
    variants::Vector{Symbol}=_parse_variants(),
    device::Symbol=_parse_device(),
    scenarios=nothing,
    want_gpu=nothing,
    weak_scale=nothing,
    weak_rows::Union{Nothing, Int}=nothing,
    weak_cols::Union{Nothing, Int}=nothing,
)
    threads = _thread_count()
    weak_rows, weak_cols, weak_scale_used = _weak_dims(rows, cols, threads; weak_scale=weak_scale, weak_rows=weak_rows, weak_cols=weak_cols)
    scenarios = _normalize_scenarios(scenarios)

    device = _resolve_device(device)
    want_gpu = want_gpu === nothing ? (get(ENV, "SEAM_GPU", "1") != "0") : Bool(want_gpu)
    if !want_gpu || device === :cpu
        if device === :cpu && any(v -> v in GPU_VARIANTS, variants)
            @warn "No GPU backend detected; skipping GPU variants. Load a backend (e.g. `using CUDA`) or set SEAM_DEVICE."
        end
        variants = filter(v -> !(v in GPU_VARIANTS), variants)
    end

    if isempty(variants)
        error("No benchmark variants selected (SEAM_VARIANTS / SEAM_GPU filtered all variants).")
    end

    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    out_dir = joinpath(RESULTS_APP_DIR, ts)
    mkpath(out_dir)

    println("="^70)
    println("SEAM-CARVING BENCHMARK (DaggerSeamCarving variants)")
    println("="^70)
    println("Threads: $threads")
    println("Runs: $runs")
    println("Strong size: $(rows)x$(cols)")
    if weak_scale_used === nothing
        println("Weak size: $(weak_rows)x$(weak_cols)")
    else
        println("Weak size: $(weak_rows)x$(weak_cols) (scale=$(round(weak_scale_used, digits=3)))")
    end
    println("Seams (k): $k")
    println("Tile size: $(tile_h)x$(tile_w)")
    println("Variants: $(join(string.(variants), ", "))")
    println()

    for scenario in scenarios
        s_rows, s_cols, label = scenario == "strong" ? (rows, cols, "Strong") : (weak_rows, weak_cols, "Weak")
        println(">>> $label scaling (rows=$(s_rows), cols=$(s_cols))")
        csv_path = joinpath(out_dir, "$(scenario)_scaling.csv")
        first = true
        for variant in variants
            times = _bench_variant(variant, s_rows, s_cols, k, tile_h, tile_w, device, runs)
            println(@sprintf("  %-24s mean=%.4fs  std=%.4fs", string(variant), mean(times), std(times; corrected=false)))
            _write_runs_csv(csv_path, scenario, variant, threads, s_rows, s_cols, k, tile_h, tile_w, times; write_header=first)
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
