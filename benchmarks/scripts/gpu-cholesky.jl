# GPU Cholesky benchmark (Dagger DArray Cholesky; 1–4 GPUs via tile assignment).
#
# Load one GPU backend **before** Dagger (see Dagger.jl docs), e.g.:
#   julia --project=apps/gpu-cholesky -e 'using CUDA; using Dagger; include("benchmarks/scripts/gpu-cholesky.jl"); run_benchmark()'
# AMD (ROCm / MI200 / MI300, etc.):
#   julia --project=apps/gpu-cholesky -e 'using AMDGPU; using Dagger; include("benchmarks/scripts/gpu-cholesky.jl"); run_benchmark()'
#
# Env:
#   CHOLESKY_NUM_GPUS    max GPUs to use for the Dagger path (default 4, clamped to visible count).
#                        Set to 1 for reduced / single-GPU runs (same `run_benchmark()` as paper tier).
#   CHOLESKY_DEVICE      auto|cuda|amdgpu|oneapi|metal (default auto)
#   CHOLESKY_K_MIN       exponent min for N=2^k (default 10 → N=1024)
#   CHOLESKY_K_MAX       exponent max (default 18 → N=262144)
#   CHOLESKY_NS          optional comma-separated N list (overrides k sweep)
#   CHOLESKY_BLOCK       tile size, power of 2 (default 512)
#   CHOLESKY_TRIALS      timing repetitions per N (default 5)
#   CHOLESKY_WARMUP      extra warmup runs (default 1)
#   CHOLESKY_ELTYPE      Float32|Float64 (default Float32)
#   CHOLESKY_CHECK       if 1 and N≤4096, run correctness check (default 0)
#   CHOLESKY_VENDOR      if 1, also time single-GPU LinearAlgebra.cholesky! (vendor potrf) on
#                        the same SPD matrix as the Dagger run (uses loaded GPU backend; default 1)
#   CHOLESKY_VENDOR_DEVICE  device ordinal for the dense vendor baseline (0-based; default 0)
#   CHOLESKY_PERF_LOG    if 1, enable Dagger TimespanLogging and append one NDJSON summary line
#                        per N to perf_dagger.jsonl (see CHOLESKY_PERF_LOG_PATH)
#   CHOLESKY_PERF_SCOPE  timed|full (default timed): timed = logs only around Dagger timing trials
#                        (after warmup); full = from matrix construction through Dagger trials
#   CHOLESKY_PERF_LOG_PATH  output file (default <results>/perf_dagger.jsonl)
#   CHOLESKY_BLOCKS      optional comma-separated tile sizes (overrides a single CHOLESKY_BLOCK).
#                        For each N, sizes that do not divide N are skipped. Example: 256,512,1024
#   CHOLESKY_INPLACE     if 1 (default), use cholesky! with copyto! from a template each rep
#                        (like the vendor path). Set to 0 for the out-of-place cholesky path.
#   CHOLESKY_ALGO        algorithm variant: rl (right-looking, original), rl_la (right-looking +
#                        processor pinning + lookahead, default), ll (left-looking + pinning).
#                        Comma-separated list sweeps algorithms: rl,rl_la,ll
#   CHOLESKY_FORCE_RL_LA if 1/true: on a single GPU, keep :rl_la instead of the default remap to :rl
#   (shell) CHOLESKY_KEEP_SYSTEM_CUDA_LD=1 — do not strip /usr/local/cuda from LD_LIBRARY_PATH in orchestrators

using BenchmarkTools
using Dates
using Dagger
using Printf
using Statistics

const APP = "gpu-cholesky"
const APP_DIR = abspath(joinpath(@__DIR__, "..", "..", "apps", APP))
const APP_IMPL = joinpath(APP_DIR, "src", "DaggerGpuCholesky.jl")
const RESULTS_APP_DIR = abspath(joinpath(@__DIR__, "..", "results", APP))

include(APP_IMPL)
if !isdefined(@__MODULE__, :DaggerGpuCholesky)
    error("DaggerGpuCholesky module not found after include($APP_IMPL).")
end
const DG = getfield(@__MODULE__, :DaggerGpuCholesky)

@inline _invoke0(f) = f()

function _bench_times(f, runs::Int)::Vector{Float64}
    f()
    trial = BenchmarkTools.@benchmark _invoke0($f) samples=runs evals=1
    return trial.times ./ 1e9
end

function _parse_int(env_key::String, default::Int)
    v = strip(get(ENV, env_key, string(default)))
    isempty(v) && return default
    return parse(Int, v)
end

function _parse_ns_list()::Union{Nothing,Vector{Int}}
    raw = strip(get(ENV, "CHOLESKY_NS", ""))
    isempty(raw) && return nothing
    parts = split(raw, r"[,\s]+", keepempty=false)
    return [parse(Int, p) for p in parts]
end

function _parse_blocks_list()::Union{Nothing,Vector{Int}}
    raw = strip(get(ENV, "CHOLESKY_BLOCKS", ""))
    isempty(raw) && return nothing
    parts = split(raw, r"[,\s]+", keepempty=false)
    return [parse(Int, p) for p in parts]
end

function _parse_algo_list()::Vector{Symbol}
    raw = strip(get(ENV, "CHOLESKY_ALGO", "rl_la"))
    parts = split(raw, r"[,\s]+", keepempty=false)
    algos = Symbol[]
    for p in parts
        s = Symbol(p)
        s in DG.CHOLESKY_ALGORITHMS ||
            error("Unknown CHOLESKY_ALGO value: $p. Use rl, rl_la, or ll.")
        push!(algos, s)
    end
    isempty(algos) && push!(algos, :rl_la)
    return algos
end

"""On a single GPU, Dagger's `:rl_la` (pinning + lookahead) can fault the CUDA stack; use `:rl` unless forced."""
function _effective_algo_list_for_gpu_count(algo_list::Vector{Symbol}, n_gpu::Int)::Vector{Symbol}
    force = strip(get(ENV, "CHOLESKY_FORCE_RL_LA", "")) in ("1", "true", "yes", "y")
    n_gpu > 1 && return algo_list
    force && return algo_list
    out = Symbol[a === :rl_la ? :rl : a for a in algo_list]
    if out != algo_list
        @warn "Single GPU: replacing algorithm :rl_la with :rl (Dagger rl_la + pinning can illegal-access CUDA). Set CHOLESKY_FORCE_RL_LA=1 to keep :rl_la." algos_in=algo_list algos_out=out
    end
    return out
end

function _parse_eltype()
    s = uppercase(strip(get(ENV, "CHOLESKY_ELTYPE", "Float32")))
    if s == "FLOAT32"
        return Float32
    elseif s == "FLOAT64"
        return Float64
    else
        error("Unknown CHOLESKY_ELTYPE: $s. Use Float32 or Float64.")
    end
end

function _parse_bool01(key::String, default::Bool)
    v = strip(get(ENV, key, default ? "1" : "0"))
    return v in ("1", "true", "yes", "y")
end

function _parse_perf_scope()::String
    s = lowercase(strip(get(ENV, "CHOLESKY_PERF_SCOPE", "timed")))
    s in ("timed", "full") || error("CHOLESKY_PERF_SCOPE must be \"timed\" or \"full\", got \"$s\"")
    return s
end

function _json_escape_string(s::AbstractString)
    io = IOBuffer()
    for c in s
        if c == '\\' || c == '"'
            print(io, '\\', c)
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\r'
            print(io, "\\r")
        elseif c == '\t'
            print(io, "\\t")
        else
            write(io, c)
        end
    end
    return String(take!(io))
end

function _json_val(v)
    if v === nothing || v === missing
        return "null"
    elseif v isa AbstractDict
        return _json_obj(Dict{String,Any}(string(k) => vv for (k, vv) in v))
    elseif v isa AbstractString || v isa Symbol
        return '"' * _json_escape_string(string(v)) * '"'
    elseif v isa Bool || v isa Integer || v isa AbstractFloat
        return string(v)
    elseif v isa AbstractVector
        return string('[', join((_json_val(x) for x in v), ','), ']')
    else
        return '"' * _json_escape_string(repr(v)) * '"'
    end
end

function _json_obj(d::Dict{String,Any})
    parts = String[]
    # Emit keys in sorted order; do not sort Pair{String,Any} (value comparison can throw).
    for k in sort(collect(keys(d)))
        v = d[k]
        push!(parts, '"' * _json_escape_string(k) * "\":" * _json_val(v))
    end
    return "{" * join(parts, ",") * "}"
end

function _perf_append_jsonl(path::AbstractString, row::Dict{String,Any})
    open(path, "a") do io
        println(io, _json_obj(row))
    end
    return path
end

function _perf_summarize_logs(logs::Dict)
    dur_s = Dict{String,Float64}()
    counts = Dict{String,Int}()
    Dagger.logs_event_pairs(logs) do w, i0, i1
        core = logs[w][:core]
        cat = core[i0].category
        key = string(cat)
        dt = (core[i1].timestamp - core[i0].timestamp) / 1e9
        dur_s[key] = get(dur_s, key, 0.0) + dt
        counts[key] = get(counts, key, 0) + 1
    end
    tfreq = Dict{String,Int}()
    for w in keys(logs)
        logs_w = logs[w]
        haskey(logs_w, :taskfuncnames) || continue
        for name in logs_w[:taskfuncnames]
            name === nothing && continue
            s = String(name)
            tfreq[s] = get(tfreq, s, 0) + 1
        end
    end
    top = if isempty(tfreq)
        Dict{String,Int}()
    else
        prs = sort(collect(tfreq), by = x -> x[2], rev = true)
        nmax = 40
        Dict(prs[1:min(nmax, length(prs))])
    end
    npw = Dict{String,Int}(string(w) => length(logs[w][:core]) for w in keys(logs))
    return Dict{String,Any}(
        "workers" => sort(collect(keys(logs))),
        "core_event_count_per_worker" => npw,
        "paired_span_duration_s_by_category" => dur_s,
        "paired_span_count_by_category" => counts,
        "taskfunc_event_top" => top,
    )
end

function _ensure_results_dir!()
    mkpath(RESULTS_APP_DIR)
    stamp = Dates.format(Dates.now(), dateformat"yyyy-mm-dd_HH-MM-SS")
    out = joinpath(RESULTS_APP_DIR, stamp)
    mkpath(out)
    return out
end

function _write_csv(path::AbstractString, header::Vector{String}, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for r in rows
            println(io, join(string.(r), ","))
        end
    end
    return path
end

"""
    run_benchmark()

Strong scaling over `N` (power-of-two grid by default):

- **Dagger:** `cholesky` on a GPU-backed `DArray` (same SPD matrix as the app README). Uses **1–4** GPUs (`CHOLESKY_NUM_GPUS`, default 4); tile assignment generalizes the paper 2×2 block-cyclic layout.
- **Vendor baseline:** by default, also times **single-GPU** `LinearAlgebra.cholesky!` (vendor **potrf**
  for CUDA / ROCm / oneAPI / Metal) on a dense GPU matrix with the **same** entries, after `copyto!`
  from a template each trial (`CHOLESKY_VENDOR=0` to disable).
Results CSV columns include `dagger_*` and `vendor_*` timings; `vendor_*` are `missing` when vendor timing is off.

With `CHOLESKY_PERF_LOG=1`, also appends one NDJSON record per `(N, block_size, algorithm)` to `perf_dagger.jsonl`.

Use `CHOLESKY_BLOCKS=a,b,...` to sweep tile sizes in one run (vendor baseline is timed **once per N**).
Use `CHOLESKY_INPLACE=1` for `cholesky!` + `copyto!` from a template each repetition (~2× `DArray` memory).
Use `CHOLESKY_ALGO=rl,rl_la,ll` to sweep algorithm variants (default `rl_la`).

On **one** GPU (`CHOLESKY_NUM_GPUS=1`), `:rl_la` is automatically mapped to `:rl` unless
`CHOLESKY_FORCE_RL_LA=1` (the rl_la path has triggered CUDA illegal memory access with a single device).
"""
function _warn_ld_library_path_cuda()
    DG.device_from_loaded() === :cuda || return nothing
    lp = get(ENV, "LD_LIBRARY_PATH", "")
    isempty(lp) && return nothing
    if occursin("/usr/local/cuda", lp)
        @warn "LD_LIBRARY_PATH contains /usr/local/cuda — mixing system CUDA with CUDA.jl often causes illegal memory access or heap corruption during GPU Cholesky. Remove those entries for this Julia process (see benchmarks/run_smoke_all.sh) or set CHOLESKY_KEEP_SYSTEM_CUDA_LD=1 only if you intend to use the system stack." maxlog = 1
    end
    return nothing
end

function run_benchmark()
    DG.resolve_device_strict()
    _warn_ld_library_path_cuda()
    outdir = _ensure_results_dir!()
    csv_path = joinpath(outdir, "cholesky_times.csv")

    k_min = _parse_int("CHOLESKY_K_MIN", 10)
    k_max = _parse_int("CHOLESKY_K_MAX", 18)
    block_size = _parse_int("CHOLESKY_BLOCK", 512)
    blocks_list = _parse_blocks_list()
    block_sizes = something(blocks_list, [block_size])
    all(bs -> bs > 0, block_sizes) || error("All CHOLESKY_BLOCKS must be positive")
    use_inplace = _parse_bool01("CHOLESKY_INPLACE", true)
    algo_list = _parse_algo_list()
    n_trials = _parse_int("CHOLESKY_TRIALS", 5)
    n_warmup = _parse_int("CHOLESKY_WARMUP", 1)
    T = _parse_eltype()
    do_check = _parse_bool01("CHOLESKY_CHECK", false)

    ns = _parse_ns_list()
    if ns === nothing
        k_min ≤ k_max || error("CHOLESKY_K_MIN ($k_min) must be ≤ CHOLESKY_K_MAX ($k_max)")
        ns = [2^k for k in k_min:k_max]
    end

    run_vendor = _parse_bool01("CHOLESKY_VENDOR", true)
    gpu_procs = DG.gpu_processors_for_cholesky()
    algo_list = _effective_algo_list_for_gpu_count(algo_list, length(gpu_procs))
    device = DG.device_from_loaded()
    vendor_dev = _parse_int("CHOLESKY_VENDOR_DEVICE", 0)

    perf_log = _parse_bool01("CHOLESKY_PERF_LOG", false)
    perf_scope = perf_log ? _parse_perf_scope() : "timed"
    perf_path =
        perf_log ? begin
            p = strip(get(ENV, "CHOLESKY_PERF_LOG_PATH", ""))
            isempty(p) ? joinpath(outdir, "perf_dagger.jsonl") : abspath(p)
        end : ""

    header = [
        "N",
        "block_size",
        "algorithm",
        "eltype",
        "device",
        "dagger_median_s",
        "dagger_mean_s",
        "dagger_std_s",
        "vendor_gpu",
        "vendor_median_s",
        "vendor_mean_s",
        "vendor_std_s",
        "trials",
        "correct_ok",
    ]
    rows = Vector{Vector{Any}}()

    @info "GPU Cholesky benchmark" outdir device n_gpu_procs=length(gpu_procs) block_sizes algo_list n_trials ns_begin=first(ns) ns_end=last(ns) run_vendor vendor_dev use_inplace perf_log perf_scope

    if perf_log
        Dagger.enable_logging!(;
            metrics = true,
            all_task_deps = false,
            tasknames = true,
            taskfuncnames = true,
        )
    end
    try
        for N in ns
            N > 0 || continue
            if !any(bs -> rem(N, bs) == 0, block_sizes)
                @warn "Skipping N=$N: no block size in $block_sizes divides N"
                continue
            end

            @info ">>> N=$N starting"; flush(stderr)
            diag_scale = T(N)
            v_gpu = missing
            v_med = missing
            v_mn = missing
            v_sd = missing
            # Vendor baseline is dense single-GPU; do not repeat when sweeping block sizes.
            # Wrapped in try/catch so an OOM or vendor error doesn't prevent the Dagger runs.
            if run_vendor
                try
                    @info "  vendor: allocating"; flush(stderr)
                    A_tpl = DG.spd_ones_dense_vendor(T, N; diag_scale=diag_scale, device_id=vendor_dev)
                    A_wrk = similar(A_tpl)
                    @info "  vendor: warmup"; flush(stderr)
                    for _ in 1:n_warmup
                        copyto!(A_wrk, A_tpl)
                        DG.bench_vendor_cholesky_once!(A_wrk)
                    end
                    @info "  vendor: timing"; flush(stderr)
                    vt = Vector{Float64}(undef, n_trials)
                    for vi in 1:n_trials
                        copyto!(A_wrk, A_tpl)
                        DG.vendor_timing_sync!()
                        t0 = time_ns()
                        DG.bench_vendor_cholesky_once!(A_wrk)
                        vt[vi] = (time_ns() - t0) / 1e9
                    end
                    v_gpu = vendor_dev
                    v_med = median(vt)
                    v_mn = mean(vt)
                    v_sd = std(vt; corrected=false)
                    DG.unsafe_free_dense_gpu!(A_wrk)
                    DG.unsafe_free_dense_gpu!(A_tpl)
                catch ex
                    @warn "Vendor baseline failed for N=$N; Dagger runs will continue" exception=(ex, catch_backtrace())
                end
                GC.gc(true)
                DG.vendor_gpu_reclaim!()
                @info "  vendor: GPU memory released"; flush(stderr)
            end

            for bs in block_sizes
                rem(N, bs) == 0 || continue
                n_blocks = N ÷ bs
                asg = DG.cholesky_tile_assignment(gpu_procs, n_blocks)

                for algo in algo_list
                @info "  dagger: N=$N bs=$bs algo=$algo"; flush(stderr)
                if perf_log && perf_scope == "full"
                    Dagger.fetch_logs!()
                end

                correct_ok = missing
                if do_check && N ≤ 4096
                    @info "  dagger: correctness check"; flush(stderr)
                    DA_chk = DG.spd_ones_darray(T, N, bs, asg)
                    correct_ok = DG.verify_cholesky_small(DA_chk; rtol=T == Float32 ? 1e-3 : 1e-8)
                    @info "correctness" N bs algo correct_ok
                    DG.unsafe_free_darray!(DA_chk)
                    DG.purge_gpu_memory!()
                end

                @info "  dagger: warmup ($n_warmup)"; flush(stderr)
                for _ in 1:n_warmup
                    DA_w = DG.spd_ones_darray(T, N, bs, asg)
                    if use_inplace
                        DA_t = copy(DA_w); DG.wait_darray!(DA_t)
                        Base.copyto!(DA_w, DA_t)
                        DG.bench_cholesky_inplace_once!(DA_w, algo)
                        DG.unsafe_free_darray!(DA_t)
                    else
                        DG.bench_cholesky_once!(DA_w, algo)
                    end
                    DG.unsafe_free_darray!(DA_w)
                    DG.purge_gpu_memory!()
                end
                DG.purge_gpu_memory!()
                @info "  dagger: warmup done, memory purged"; flush(stderr)

                if perf_log && perf_scope == "timed"
                    Dagger.fetch_logs!()
                end

                times = Vector{Float64}(undef, n_trials)
                for trial_i in 1:n_trials
                    DA_w = DG.spd_ones_darray(T, N, bs, asg)
                    if use_inplace
                        DA_t = copy(DA_w); DG.wait_darray!(DA_t)
                        Base.copyto!(DA_w, DA_t)
                        DG.purge_gpu_memory!()
                        t0 = time_ns()
                        DG.bench_cholesky_inplace_once!(DA_w, algo)
                        times[trial_i] = (time_ns() - t0) / 1e9
                        DG.unsafe_free_darray!(DA_t)
                    else
                        DG.purge_gpu_memory!()
                        t0 = time_ns()
                        DG.bench_cholesky_once!(DA_w, algo)
                        times[trial_i] = (time_ns() - t0) / 1e9
                    end
                    DG.unsafe_free_darray!(DA_w)
                    DG.purge_gpu_memory!()
                end
                perf_logs = perf_log ? Dagger.fetch_logs!() : nothing
                d_med = median(times)
                d_mn = mean(times)
                d_sd = std(times; corrected=false)

                push!(
                    rows,
                    Any[
                        N,
                        bs,
                        string(algo),
                        string(T),
                        string(device),
                        d_med,
                        d_mn,
                        d_sd,
                        v_gpu,
                        v_med,
                        v_mn,
                        v_sd,
                        n_trials,
                        correct_ok,
                    ],
                )
                if run_vendor
                    @printf(
                        stderr,
                        "N=%7d bs=%4d algo=%-5s  Dagger median=%.4fs  vendor median=%.4fs  (GPU %d)\n",
                        N,
                        bs,
                        string(algo),
                        d_med,
                        v_med,
                        vendor_dev,
                    )
                else
                    @printf(stderr, "N=%7d bs=%4d algo=%-5s  Dagger median=%.4fs  mean=%.4fs\n", N, bs, string(algo), d_med, d_mn)
                end

                if perf_log
                    summary = _perf_summarize_logs(perf_logs)
                    row = Dict{String,Any}(
                        "kind" => "gpu_cholesky_perf",
                        "results_stamp" => basename(outdir),
                        "scope" => perf_scope,
                        "N" => N,
                        "block_size" => bs,
                        "algorithm" => string(algo),
                        "inplace" => use_inplace,
                        "eltype" => string(T),
                        "device" => string(device),
                        "dagger_median_s" => d_med,
                        "dagger_mean_s" => d_mn,
                        "dagger_std_s" => d_sd,
                        "trials" => n_trials,
                        "summary" => summary,
                    )
                    if run_vendor
                        row["vendor_median_s"] = v_med
                        row["vendor_device"] = vendor_dev
                    end
                    _perf_append_jsonl(perf_path, row)
                end
                end # algo
            end
        end
    finally
        if perf_log
            Dagger.disable_logging!()
        end
    end

    _write_csv(csv_path, header, rows)
    @info "Wrote" csv_path
    if perf_log
        @info "Dagger perf JSONL" perf_path
    end
    return csv_path
end
