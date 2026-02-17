using BenchmarkTools
using Dates
using Dagger
using Printf
using Statistics

const APP = "heat-propagation"
const APP_DIR = abspath(joinpath(@__DIR__, "..", "..", "apps", APP))
const APP_IMPL = joinpath(APP_DIR, "src", "DaggerHeatPropagation.jl")
const RESULTS_APP_DIR = abspath(joinpath(@__DIR__, "..", "results", APP))
const DEVICE_BACKENDS = (
    (:cuda, :CUDA, :CuArray),
    (:amdgpu, :AMDGPU, :ROCArray),
    (:oneapi, :oneAPI, :oneArray),
    (:metal, :Metal, :MtlArray),
)
const DEFAULT_GPU_SIZES = Int[1024, 2048, 4096, 8192, 12288, 16384]

include(APP_IMPL)
if !isdefined(@__MODULE__, :DaggerHeatPropagation)
    error("DaggerHeatPropagation module not found after include($APP_IMPL).")
end
const DaggerHeatPropagation = getfield(@__MODULE__, :DaggerHeatPropagation)

const DEFAULT_VARIANTS = [
    :cpu_serial_pad,
    :cpu_dagger_stencil_pad,
    :cpu_dagger_stencil_wrap,
    :gpu_dagger_stencil_pad,
    :gpu_dagger_stencil_wrap,
]
const GPU_VARIANTS = Set([:gpu_dagger_stencil_pad, :gpu_dagger_stencil_wrap])

function _parse_device()
    raw = lowercase(strip(get(ENV, "HEAT_DEVICE", "auto")))
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
        error("Unknown HEAT_DEVICE: $raw. Use auto|cpu|cuda|amdgpu|oneapi|metal.")
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

_resolve_device(device::Symbol) = device === :auto ? _device_from_loaded() : device

function _device_convert(device::Symbol, A::AbstractArray)
    device = _resolve_device(device)
    if device === :cpu
        return A
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
            return getfield(mod, ctor)(A)
        end
    end
    error("Unknown device: $device")
end

function _parse_variants()
    raw = strip(get(ENV, "HEAT_VARIANTS", ""))
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
        error("Unknown HEAT_VARIANTS: $(join(string.(unknown), ", ")). Known: $(join(string.(DEFAULT_VARIANTS), ", "))")
    end
    return variants
end

_thread_count() = Threads.nthreads()
@inline _invoke0(f) = f()

function _bench_times(f, runs::Int)::Vector{Float64}
    f() # warmup (compile)
    trial = BenchmarkTools.@benchmark _invoke0($f) samples=runs evals=1
    return trial.times ./ 1e9
end

function _wait_darray!(A::Dagger.DArray)
    for chunk in A.chunks
        fetch(chunk)
    end
    return nothing
end

function _build_initial(rows::Int, cols::Int, ambient::Float64, hotspot_temp::Float64, hotspot_radius::Int)
    plate = DaggerHeatPropagation.ambient_plate(rows, cols; ambient=ambient)

    DaggerHeatPropagation.add_hotspot!(plate;
        row=cld(rows, 3),
        col=cld(cols, 3),
        radius=hotspot_radius,
        temperature=hotspot_temp,
    )

    DaggerHeatPropagation.add_hotspot!(plate;
        row=cld(2 * rows, 3),
        col=cld(2 * cols, 3),
        radius=max(2, hotspot_radius ÷ 2),
        temperature=0.75 * hotspot_temp,
    )

    DaggerHeatPropagation.add_gaussian_hotspot!(plate;
        row=cld(rows, 2),
        col=cld(cols, 2),
        sigma=max(3.0, 1.5 * hotspot_radius),
        amplitude=0.50 * hotspot_temp,
    )

    return plate
end

function _run_variant(
    variant::Symbol,
    initial::AbstractMatrix{<:Real},
    steps::Int,
    alpha::Float64,
    block_h::Int,
    block_w::Int,
    pad_value::Float64,
    device::Symbol,
)
    if variant === :cpu_serial_pad
        return DaggerHeatPropagation.heat_propagate_cpu_serial(initial;
            steps=steps,
            alpha=alpha,
            boundary=:pad,
            pad_value=pad_value,
        )
    elseif variant === :cpu_dagger_stencil_pad
        return DaggerHeatPropagation.heat_propagate_dagger_stencil(initial;
            steps=steps,
            alpha=alpha,
            block_h=block_h,
            block_w=block_w,
            boundary=:pad,
            pad_value=pad_value,
        )
    elseif variant === :cpu_dagger_stencil_wrap
        return DaggerHeatPropagation.heat_propagate_dagger_stencil(initial;
            steps=steps,
            alpha=alpha,
            block_h=block_h,
            block_w=block_w,
            boundary=:wrap,
            pad_value=pad_value,
        )
    elseif variant === :gpu_dagger_stencil_pad || variant === :gpu_dagger_stencil_wrap
        resolved = _resolve_device(device)
        resolved === :cpu && error("GPU variant requested ($variant), but no GPU backend is loaded.")
        dev_initial = _device_convert(resolved, Float32.(initial))
        tiles = DArray(dev_initial, Blocks(block_h, block_w))
        if variant === :gpu_dagger_stencil_pad
            return DaggerHeatPropagation.heat_propagate_dagger_stencil(tiles;
                steps=steps,
                alpha=alpha,
                boundary=:pad,
                pad_value=pad_value,
                return_darray=true,
            )
        else
            return DaggerHeatPropagation.heat_propagate_dagger_stencil(tiles;
                steps=steps,
                alpha=alpha,
                boundary=:wrap,
                pad_value=pad_value,
                return_darray=true,
            )
        end
    else
        error("Unknown heat-propagation variant: $variant")
    end
end

function heat_job(
    variant::Symbol,
    rows::Int,
    cols::Int,
    steps::Int,
    alpha::Float64,
    block_h::Int,
    block_w::Int,
    ambient::Float64,
    hotspot_temp::Float64,
    hotspot_radius::Int,
    pad_value::Float64,
    device::Symbol,
)
    initial = _build_initial(rows, cols, ambient, hotspot_temp, hotspot_radius)
    out = _run_variant(variant, initial, steps, alpha, block_h, block_w, pad_value, device)
    if out isa Dagger.DArray
        _wait_darray!(out)
        return Float64(size(out, 1) * size(out, 2))
    end
    return Float64(sum(out))
end

function _write_runs_csv(path::AbstractString, scenario::AbstractString, variant::Symbol, device::Symbol, threads::Int,
                         rows::Int, cols::Int, steps::Int, block_h::Int, block_w::Int,
                         alpha::Float64, ambient::Float64, hotspot_temp::Float64, hotspot_radius::Int,
                         pad_value::Float64, times::Vector{Float64}; write_header::Bool=false)
    open(path, write_header ? "w" : "a") do io
        if write_header
            println(io, "scenario,variant,device,threads,rows,cols,steps,block_h,block_w,alpha,ambient,hotspot_temp,hotspot_radius,pad_value,run,time_sec")
        end
        alpha_s = @sprintf("%.6f", alpha)
        ambient_s = @sprintf("%.6f", ambient)
        hotspot_temp_s = @sprintf("%.6f", hotspot_temp)
        pad_value_s = @sprintf("%.6f", pad_value)
        for (i, t) in enumerate(times)
            time_s = @sprintf("%.9f", t)
            println(io, "$(scenario),$(variant),$(device),$(threads),$(rows),$(cols),$(steps),$(block_h),$(block_w),$(alpha_s),$(ambient_s),$(hotspot_temp_s),$(hotspot_radius),$(pad_value_s),$(i),$(time_s)")
        end
    end
end

function _bench_variant(variant::Symbol, rows::Int, cols::Int, steps::Int, alpha::Float64,
                        block_h::Int, block_w::Int, ambient::Float64,
                        hotspot_temp::Float64, hotspot_radius::Int,
                        pad_value::Float64, device::Symbol, runs::Int)
    f = () -> heat_job(variant, rows, cols, steps, alpha, block_h, block_w, ambient, hotspot_temp, hotspot_radius, pad_value, device)
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

    if haskey(ENV, "HEAT_WEAK_ROWS") || haskey(ENV, "HEAT_WEAK_COLS")
        wrows = parse(Int, get(ENV, "HEAT_WEAK_ROWS", string(rows)))
        wcols = parse(Int, get(ENV, "HEAT_WEAK_COLS", string(cols)))
        return wrows, wcols, nothing
    end

    if weak_scale === nothing
        weak_scale = lowercase(strip(get(ENV, "HEAT_WEAK_SCALE", "sqrt")))
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
    raw = lowercase(strip(get(ENV, "HEAT_SCENARIOS", "both")))
    if raw in ("both", "all", "")
        return ("strong", "weak")
    elseif raw in ("strong",)
        return ("strong",)
    elseif raw in ("weak",)
        return ("weak",)
    else
        error("Unknown HEAT_SCENARIOS=$raw. Use strong|weak|both.")
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

function _env_flag(name::AbstractString, default::Bool)::Bool
    raw = lowercase(strip(get(ENV, name, default ? "1" : "0")))
    return raw in ("1", "true", "yes", "y", "on")
end

function _parse_size_list(raw::AbstractString)
    tokens = split(raw, r"[,\s]+", keepempty=false)
    isempty(tokens) && error("GPU size list is empty.")
    sizes = Int[]
    for token in tokens
        n = parse(Int, token)
        n > 0 || error("GPU size must be > 0, got $n")
        push!(sizes, n)
    end
    return sizes
end

function _parse_gpu_sizes()
    raw = strip(get(ENV, "HEAT_GPU_SIZES", ""))
    return isempty(raw) ? copy(DEFAULT_GPU_SIZES) : _parse_size_list(raw)
end

function _parse_gpu_variant()
    raw = lowercase(strip(get(ENV, "HEAT_GPU_VARIANT", "gpu_dagger_stencil_pad")))
    variant = Symbol(raw)
    variant in GPU_VARIANTS || error("Unknown HEAT_GPU_VARIANT=$raw. Use one of: $(join(string.(collect(GPU_VARIANTS)), ", ")).")
    return variant
end

function _write_gpu_size_runs_csv(path::AbstractString, variant::Symbol, device::Symbol, threads::Int,
                                  size::Int, steps::Int, block_h::Int, block_w::Int, alpha::Float64,
                                  ambient::Float64, hotspot_temp::Float64, hotspot_radius::Int,
                                  pad_value::Float64, times::Vector{Float64}; write_header::Bool=false)
    open(path, write_header ? "w" : "a") do io
        if write_header
            println(io, "variant,device,threads,size,rows,cols,steps,block_h,block_w,alpha,ambient,hotspot_temp,hotspot_radius,pad_value,run,time_sec")
        end
        alpha_s = @sprintf("%.6f", alpha)
        ambient_s = @sprintf("%.6f", ambient)
        hotspot_temp_s = @sprintf("%.6f", hotspot_temp)
        pad_value_s = @sprintf("%.6f", pad_value)
        for (i, t) in enumerate(times)
            println(io, "$(variant),$(device),$(threads),$(size),$(size),$(size),$(steps),$(block_h),$(block_w),$(alpha_s),$(ambient_s),$(hotspot_temp_s),$(hotspot_radius),$(pad_value_s),$(i),$(@sprintf("%.9f", t))")
        end
    end
end

function _write_gpu_size_summary_csv(path::AbstractString, variant::Symbol, device::Symbol, threads::Int,
                                     size::Int, steps::Int, block_h::Int, block_w::Int, alpha::Float64,
                                     ambient::Float64, hotspot_temp::Float64, hotspot_radius::Int,
                                     pad_value::Float64, times::Vector{Float64}; write_header::Bool=false)
    open(path, write_header ? "w" : "a") do io
        if write_header
            println(io, "variant,device,threads,size,rows,cols,steps,block_h,block_w,alpha,ambient,hotspot_temp,hotspot_radius,pad_value,mean_sec,std_sec")
        end
        alpha_s = @sprintf("%.6f", alpha)
        ambient_s = @sprintf("%.6f", ambient)
        hotspot_temp_s = @sprintf("%.6f", hotspot_temp)
        pad_value_s = @sprintf("%.6f", pad_value)
        mean_s = @sprintf("%.9f", mean(times))
        std_s = @sprintf("%.9f", std(times; corrected=false))
        println(io, "$(variant),$(device),$(threads),$(size),$(size),$(size),$(steps),$(block_h),$(block_w),$(alpha_s),$(ambient_s),$(hotspot_temp_s),$(hotspot_radius),$(pad_value_s),$(mean_s),$(std_s)")
    end
end

function _ensure_plots()
    if isdefined(@__MODULE__, :Plots)
        return Base.invokelatest(() -> getfield(@__MODULE__, :Plots))
    end
    try
        @eval import Plots
    catch err
        msg = sprint(showerror, err)
        error("Plots.jl is required to generate plots. Run with `--project=benchmarks/scripts` or add Plots to your active project. Original error: $msg")
    end
    return Base.invokelatest(() -> getfield(@__MODULE__, :Plots))
end

function _load_gpu_size_summary(path::AbstractString)
    sizes = Int[]
    means = Float64[]
    stds = Float64[]
    first = true
    for line in eachline(path)
        if first
            first = false
            continue
        end
        s = strip(line)
        isempty(s) && continue
        cols = split(s, ",")
        length(cols) == 16 || error("Unexpected summary CSV format in $path")
        push!(sizes, parse(Int, cols[4]))
        push!(means, parse(Float64, cols[15]))
        push!(stds, parse(Float64, cols[16]))
    end
    perm = sortperm(sizes)
    return sizes[perm], means[perm], stds[perm]
end

function plot_gpu_size_sweep(summary_csv::AbstractString;
    out_png::Union{Nothing, AbstractString}=nothing,
    title::AbstractString="Heat propagation GPU size sweep",
)
    isfile(summary_csv) || error("Summary CSV not found: $summary_csv")
    out_png = out_png === nothing ? replace(summary_csv, ".csv" => ".png") : String(out_png)
    plots_mod = _ensure_plots()
    sizes, means, stds = _load_gpu_size_summary(summary_csv)
    p = Base.invokelatest(plots_mod.plot, sizes, means;
        marker=:circle,
        linewidth=2,
        ribbon=stds,
        fillalpha=0.15,
        xlabel="Matrix size N (NxN)",
        ylabel="Time (s)",
        title=title,
        legend=false,
        xscale=:log10,
        xticks=sizes,
    )
    Base.invokelatest(plots_mod.savefig, p, out_png)
    return out_png
end

"""
    run_gpu_size_sweep(; sizes=_parse_gpu_sizes(), runs=3, steps=200, alpha=0.2,
                        block_h=128, block_w=128, ambient=0.0, hotspot_temp=1.0,
                        hotspot_radius=8, pad_value=ambient, variant=_parse_gpu_variant(),
                        device=_parse_device(), make_plot=true)

Runs a GPU-only matrix-size sweep for heat propagation and writes:
- `gpu_size_sweep_runs.csv` (per-run timings)
- `gpu_size_sweep_summary.csv` (mean/std per size)
- `gpu_size_sweep.png` (optional)
"""
function run_gpu_size_sweep(;
    sizes::Vector{Int}=_parse_gpu_sizes(),
    runs::Int=parse(Int, get(ENV, "BENCH_RUNS", "3")),
    steps::Int=parse(Int, get(ENV, "HEAT_GPU_STEPS", get(ENV, "HEAT_STEPS", "200"))),
    alpha::Float64=parse(Float64, get(ENV, "HEAT_GPU_ALPHA", get(ENV, "HEAT_ALPHA", "0.2"))),
    block_h::Int=parse(Int, get(ENV, "HEAT_GPU_BLOCK_H", get(ENV, "HEAT_BLOCK_H", "128"))),
    block_w::Int=parse(Int, get(ENV, "HEAT_GPU_BLOCK_W", get(ENV, "HEAT_BLOCK_W", "128"))),
    ambient::Float64=parse(Float64, get(ENV, "HEAT_GPU_AMBIENT", get(ENV, "HEAT_AMBIENT", "0.0"))),
    hotspot_temp::Float64=parse(Float64, get(ENV, "HEAT_GPU_HOTSPOT_TEMP", get(ENV, "HEAT_HOTSPOT_TEMP", "1.0"))),
    hotspot_radius::Int=parse(Int, get(ENV, "HEAT_GPU_HOTSPOT_RADIUS", get(ENV, "HEAT_HOTSPOT_RADIUS", "8"))),
    pad_value=nothing,
    variant::Symbol=_parse_gpu_variant(),
    device::Symbol=_parse_device(),
    make_plot::Bool=_env_flag("HEAT_GPU_PLOT", true),
)
    runs > 0 || throw(ArgumentError("runs must be > 0"))
    steps >= 0 || throw(ArgumentError("steps must be >= 0"))
    0.0 < alpha <= 0.25 || throw(ArgumentError("alpha must be in (0, 0.25]"))
    block_h > 0 || throw(ArgumentError("block_h must be > 0"))
    block_w > 0 || throw(ArgumentError("block_w must be > 0"))
    hotspot_radius >= 0 || throw(ArgumentError("hotspot_radius must be >= 0"))
    variant in GPU_VARIANTS || throw(ArgumentError("variant must be GPU variant, got $variant"))
    isempty(sizes) && throw(ArgumentError("sizes cannot be empty"))
    all(>(0), sizes) || throw(ArgumentError("all sizes must be > 0"))

    pad_value = pad_value === nothing ? parse(Float64, get(ENV, "HEAT_GPU_PAD_VALUE", string(ambient))) : Float64(pad_value)
    resolved_device = _resolve_device(device)
    resolved_device === :cpu && error("No GPU backend detected. Load a backend (CUDA/AMDGPU/oneAPI/Metal) or set HEAT_DEVICE.")

    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    out_dir = joinpath(RESULTS_APP_DIR, ts, "gpu_size_sweep")
    mkpath(out_dir)
    runs_csv = joinpath(out_dir, "gpu_size_sweep_runs.csv")
    summary_csv = joinpath(out_dir, "gpu_size_sweep_summary.csv")
    threads = _thread_count()

    println("="^70)
    println("HEAT-PROPAGATION GPU SIZE SWEEP")
    println("="^70)
    println("Device: $resolved_device")
    println("Variant: $variant")
    println("Threads: $threads")
    println("Sizes: $(join(string.(sizes), ", "))")
    println("Runs: $runs")
    println("Steps: $steps")
    println("Alpha: $alpha")
    println("Blocks: $(block_h)x$(block_w)")
    println("Ambient: $ambient")
    println("Hotspot temp/radius: $(hotspot_temp) / $(hotspot_radius)")
    println("Pad value: $pad_value")
    println()

    first = true
    for size in sizes
        times = _bench_variant(variant, size, size, steps, alpha, block_h, block_w,
                               ambient, hotspot_temp, hotspot_radius, pad_value, resolved_device, runs)
        println(@sprintf("  N=%-6d mean=%.4fs  std=%.4fs", size, mean(times), std(times; corrected=false)))
        _write_gpu_size_runs_csv(runs_csv, variant, resolved_device, threads, size, steps, block_h, block_w,
                                 alpha, ambient, hotspot_temp, hotspot_radius, pad_value, times; write_header=first)
        _write_gpu_size_summary_csv(summary_csv, variant, resolved_device, threads, size, steps, block_h, block_w,
                                    alpha, ambient, hotspot_temp, hotspot_radius, pad_value, times; write_header=first)
        first = false
    end
    println()

    plot_path = nothing
    if make_plot
        plot_path = joinpath(out_dir, "gpu_size_sweep.png")
        plot_gpu_size_sweep(summary_csv; out_png=plot_path, title="Heat propagation GPU size sweep ($variant, $resolved_device)")
        println("Plot written to: $plot_path")
    end
    println("Run timings written to: $runs_csv")
    println("Summary written to: $summary_csv")
    return (; out_dir, runs_csv, summary_csv, plot_path)
end

"""
    run_benchmark(; runs=3, rows=512, cols=512, steps=200, alpha=0.2,
                  block_h=128, block_w=128, ambient=0.0,
                  hotspot_temp=1.0, hotspot_radius=8,
                  pad_value=ambient, variants=_parse_variants())

Runs strong- and weak-scaling benchmarks for heat-propagation variants.

Configuration (environment variables):
- `BENCH_RUNS` (default: 3)
- `HEAT_ROWS` (default: 512)
- `HEAT_COLS` (default: 512)
- `HEAT_STEPS` (default: 200)
- `HEAT_ALPHA` (default: 0.2)
- `HEAT_BLOCK_H` (default: 128)
- `HEAT_BLOCK_W` (default: 128)
- `HEAT_AMBIENT` (default: 0.0)
- `HEAT_HOTSPOT_TEMP` (default: 1.0)
- `HEAT_HOTSPOT_RADIUS` (default: 8)
- `HEAT_PAD_VALUE` (default: `HEAT_AMBIENT`)
- `HEAT_VARIANTS` (default: all; comma/space-separated list)
- `HEAT_GPU` (default: 1; set to 0 to skip GPU variants)
- `HEAT_DEVICE` (default: auto; cpu|cuda|amdgpu|oneapi|metal)
- `HEAT_WEAK_SCALE` (default: sqrt; options: sqrt|linear|<float>)
- `HEAT_WEAK_ROWS` / `HEAT_WEAK_COLS` (override weak dimensions)
- `HEAT_SCENARIOS` (default: both; strong|weak|both)
"""
function run_benchmark(;
    runs::Int=parse(Int, get(ENV, "BENCH_RUNS", "3")),
    rows::Int=parse(Int, get(ENV, "HEAT_ROWS", "512")),
    cols::Int=parse(Int, get(ENV, "HEAT_COLS", "512")),
    steps::Int=parse(Int, get(ENV, "HEAT_STEPS", "200")),
    alpha::Float64=parse(Float64, get(ENV, "HEAT_ALPHA", "0.2")),
    block_h::Int=parse(Int, get(ENV, "HEAT_BLOCK_H", "128")),
    block_w::Int=parse(Int, get(ENV, "HEAT_BLOCK_W", "128")),
    ambient::Float64=parse(Float64, get(ENV, "HEAT_AMBIENT", "0.0")),
    hotspot_temp::Float64=parse(Float64, get(ENV, "HEAT_HOTSPOT_TEMP", "1.0")),
    hotspot_radius::Int=parse(Int, get(ENV, "HEAT_HOTSPOT_RADIUS", "8")),
    pad_value=nothing,
    variants::Vector{Symbol}=_parse_variants(),
    device::Symbol=_parse_device(),
    scenarios=nothing,
    want_gpu=nothing,
    weak_scale=nothing,
    weak_rows::Union{Nothing, Int}=nothing,
    weak_cols::Union{Nothing, Int}=nothing,
)
    rows > 0 || throw(ArgumentError("rows must be > 0"))
    cols > 0 || throw(ArgumentError("cols must be > 0"))
    steps >= 0 || throw(ArgumentError("steps must be >= 0"))
    0.0 < alpha <= 0.25 || throw(ArgumentError("alpha must be in (0, 0.25]"))
    block_h > 0 || throw(ArgumentError("block_h must be > 0"))
    block_w > 0 || throw(ArgumentError("block_w must be > 0"))
    hotspot_radius >= 0 || throw(ArgumentError("hotspot_radius must be >= 0"))

    pad_value = pad_value === nothing ? parse(Float64, get(ENV, "HEAT_PAD_VALUE", string(ambient))) : Float64(pad_value)

    threads = _thread_count()
    weak_rows, weak_cols, weak_scale_used = _weak_dims(rows, cols, threads;
        weak_scale=weak_scale,
        weak_rows=weak_rows,
        weak_cols=weak_cols,
    )
    scenarios = _normalize_scenarios(scenarios)
    device = _resolve_device(device)

    want_gpu = want_gpu === nothing ? _env_flag("HEAT_GPU", true) : Bool(want_gpu)
    if !want_gpu || device === :cpu
        if device === :cpu && any(v -> v in GPU_VARIANTS, variants)
            @warn "No GPU backend detected; skipping GPU variants. Load a backend module (CUDA/AMDGPU/oneAPI/Metal) or set HEAT_DEVICE."
        end
        variants = filter(v -> !(v in GPU_VARIANTS), variants)
    end
    isempty(variants) && error("No benchmark variants selected (HEAT_VARIANTS / HEAT_GPU filtered all variants).")

    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    out_dir = joinpath(RESULTS_APP_DIR, ts)
    mkpath(out_dir)

    println("="^70)
    println("HEAT-PROPAGATION BENCHMARK (DaggerHeatPropagation variants)")
    println("="^70)
    println("Device: $device")
    println("Threads: $threads")
    println("Runs: $runs")
    println("Strong size: $(rows)x$(cols)")
    if weak_scale_used === nothing
        println("Weak size: $(weak_rows)x$(weak_cols)")
    else
        println("Weak size: $(weak_rows)x$(weak_cols) (scale=$(round(weak_scale_used, digits=3)))")
    end
    println("Steps: $steps")
    println("Alpha: $alpha")
    println("Blocks: $(block_h)x$(block_w)")
    println("Ambient: $ambient")
    println("Hotspot temp/radius: $(hotspot_temp) / $(hotspot_radius)")
    println("Pad value: $pad_value")
    println("Variants: $(join(string.(variants), ", "))")
    println()

    for scenario in scenarios
        s_rows, s_cols, label = scenario == "strong" ? (rows, cols, "Strong") : (weak_rows, weak_cols, "Weak")
        println(">>> $label scaling (rows=$(s_rows), cols=$(s_cols))")
        csv_path = joinpath(out_dir, "$(scenario)_scaling.csv")
        first = true
        for variant in variants
            times = _bench_variant(variant, s_rows, s_cols, steps, alpha, block_h, block_w,
                                   ambient, hotspot_temp, hotspot_radius, pad_value, device, runs)
            println(@sprintf("  %-28s mean=%.4fs  std=%.4fs", string(variant), mean(times), std(times; corrected=false)))
            _write_runs_csv(csv_path, scenario, variant, device, threads, s_rows, s_cols,
                            steps, block_h, block_w, alpha, ambient, hotspot_temp,
                            hotspot_radius, pad_value, times; write_header=first)
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
