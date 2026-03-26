module DaggerSeamCarving

using Dagger
using KernelAbstractions
using ThreadPinning

const DEFAULT_THREADS = 256

# Concurrent DP tasks in the current `_run_task_waves!` batch (wavefront/triangles). Used to cap inner
# `Threads.@threads` so nested parallelism does not oversubscribe Julia's single thread pool.
const _seam_dp_concurrent_outer_tasks = Ref{Int}(1)

@inline clampi(x, lo, hi) = x < lo ? lo : (x > hi ? hi : x)

# #region agent log
const _SEAM_AGENT_LOG = "/flare/dagger/paper/.cursor/debug-d35653.log"

function _agent_debug_log(; hypothesisId::String = "", location::String = "", message::String = "",
        data::Dict{String,Any} = Dict{String,Any}())
    open(_SEAM_AGENT_LOG, "a") do io
        print(io, "{\"sessionId\":\"d35653\",\"timestamp\":", round(Int, time() * 1000),
            ",\"hypothesisId\":", repr(hypothesisId),
            ",\"location\":", repr(location),
            ",\"message\":", repr(message), ",\"data\":{")
        first = true
        for (k, v) in data
            if !first
                print(io, ",")
            end
            first = false
            print(io, repr(k), ":")
            if v isa AbstractFloat
                print(io, v)
            elseif v isa Integer || v isa Bool
                print(io, v)
            else
                print(io, repr(string(v)))
            end
        end
        println(io, "}}")
    end
    return nothing
end
# #endregion

@inline function _seam_perf_log_enabled()::Bool
    d = lowercase(strip(get(ENV, "SEAM_DEBUG_AGENT", "")))
    p = lowercase(strip(get(ENV, "SEAM_PERF_LOG", "")))
    return d in ("1", "true", "yes", "on") || p in ("1", "true", "yes", "on")
end

function _perf_env_snapshot()::Dict{String,String}
    keys = (
        "SEAM_WAVE_SIZE", "SEAM_WESTRICK_BLOCK_WIDTH", "SEAM_WESTRICK_BLOCK_WIDTH_AUTO",
        "SEAM_WESTRICK_BLOCK_WIDTH_AUTO_MAX", "SEAM_WESTRICK_BLOCK_WIDTH_AUTO_TASK_TARGET",
        "SEAM_WESTRICK_BLOCK_WIDTH_SCHED_THREADS",
        "SEAM_WESTRICK_PHASE_TASKS_GE_THREADS",
        "SEAM_PIN_THREADS", "SEAM_TILE_H", "SEAM_TILE_W",
    )
    out = Dict{String,String}()
    for k in keys
        if haskey(ENV, k)
            out[k] = ENV[k]
        end
    end
    return out
end

"""
    seam_configure_thread_pinning!() -> Bool

If environment variable `SEAM_PIN_THREADS` is unset, empty, or `0`/`false`/`no`/`off`, does nothing and returns `false`.

Otherwise pins Julia’s threadpool using [ThreadPinning.jl](https://github.com/JuliaParallel/ThreadPinning.jl) so worker threads stay on chosen CPUs (reduces migration inside the allowed cpuset). Call **after** starting Julia with the desired `-t` / `JULIA_NUM_THREADS`. Combine with **`numactl`** / PBS cpu sets so the process mask matches the CPUs you pin to.

**Values** (case-insensitive):

| `SEAM_PIN_THREADS` | `pinthreads` policy |
|--------------------|---------------------|
| `1`, `true`, `yes`, `on`, `compact`, `threads`, `cputhreads` | `:compact` |
| `cores` | `:cores` (prefer physical cores) |
| `numa`, `numas` | `:numa` |
| `sockets` | `:sockets` |
| `firstn` | `:firstn` |
| `random` | `:random` |
| `affinitymask` | `:affinitymask` (respect current mask) |
| `current` | `:current` |

Returns `true` if pinning ran.
"""
function seam_configure_thread_pinning!()::Bool
    raw = lowercase(strip(get(ENV, "SEAM_PIN_THREADS", "")))
    if raw in ("", "0", "false", "no", "off")
        return false
    end
    sym = if raw in ("1", "true", "yes", "on", "compact", "threads", "cputhreads")
        :compact
    elseif raw == "cores"
        :cores
    elseif raw in ("numa", "numas")
        :numa
    elseif raw == "sockets"
        :sockets
    elseif raw == "firstn"
        :firstn
    elseif raw == "random"
        :random
    elseif raw == "affinitymask"
        :affinitymask
    elseif raw == "current"
        :current
    else
        error("Unknown SEAM_PIN_THREADS=$(repr(raw)). Use compact|cores|numa|sockets|firstn|affinitymask|current|0.")
    end
    try
        pinthreads(sym)
    catch err
        @warn "Thread pinning failed; continuing without pinning (common under PBS/cgroup if policy uses CPUs outside the job mask). Try SEAM_PIN_THREADS=affinitymask after numactl, or unset SEAM_PIN_THREADS." exception = err
        return false
    end
    return true
end

function energy_cpu(img::AbstractMatrix{T}) where T
    H, W = size(img)
    ET = promote_type(T, Float32)
    E = Array{ET}(undef, H, W)
    Threads.@threads for y in 1:H
        ym = y == 1 ? 1 : y - 1
        yp = y == H ? H : y + 1
        @inbounds for x in 1:W
            xm = x == 1 ? 1 : x - 1
            xp = x == W ? W : x + 1
            dx = abs(ET(img[y, xp]) - ET(img[y, xm]))
            dy = abs(ET(img[yp, x]) - ET(img[ym, x]))
            E[y, x] = dx + dy
        end
    end
    return E
end

function cumulative_energy_cpu(E::AbstractMatrix{T}) where T
    H, W = size(E)
    M = Array{T}(undef, H, W)
    B = Array{Int8}(undef, H, W)
    @inbounds begin
        M[1, :] .= E[1, :]
        B[1, :] .= 0
    end
    for y in 2:H
        Threads.@threads for x in 1:W
            @inbounds begin
                left = x > 1 ? M[y - 1, x - 1] : typemax(T)
                mid = M[y - 1, x]
                right = x < W ? M[y - 1, x + 1] : typemax(T)
                if left <= mid && left <= right
                    M[y, x] = E[y, x] + left
                    B[y, x] = -1
                elseif mid <= right
                    M[y, x] = E[y, x] + mid
                    B[y, x] = 0
                else
                    M[y, x] = E[y, x] + right
                    B[y, x] = 1
                end
            end
        end
    end
    return M, B
end

function dp_tile!(E::AbstractMatrix{T}, M::AbstractMatrix{T}, B::AbstractMatrix{Int8},
                  y1::Int, y2::Int, x1::Int, x2::Int) where T
    H, W = size(E)
    @inbounds for y in y1:y2
        if y == 1
            for x in x1:x2
                M[1, x] = E[1, x]
                B[1, x] = 0
            end
            continue
        end
        for x in x1:x2
            left = x > 1 ? M[y - 1, x - 1] : typemax(T)
            mid = M[y - 1, x]
            right = x < W ? M[y - 1, x + 1] : typemax(T)
            if left <= mid && left <= right
                M[y, x] = E[y, x] + left
                B[y, x] = -1
            elseif mid <= right
                M[y, x] = E[y, x] + mid
                B[y, x] = 0
            else
                M[y, x] = E[y, x] + right
                B[y, x] = 1
            end
        end
    end
    return nothing
end

@inline function dp_tile_wait(deps::Tuple, E, M, B, y1::Int, y2::Int, x1::Int, x2::Int)
    foreach(Dagger.fetch, deps)
    dp_tile!(E, M, B, y1, y2, x1, x2)
    return nothing
end

@inline function dp_row_tile_wait(deps::Tuple, E, M, B, y::Int, x1::Int, x2::Int)
    foreach(Dagger.fetch, deps)
    dp_row_range!(E, M, B, y, x1, x2)
    return nothing
end

@inline function _parse_wave_size(raw::AbstractString)
    v = parse(Int, strip(raw))
    v > 0 || error("SEAM_WAVE_SIZE must be a positive integer.")
    return v
end

# If true, wavefront/triangles use threaded energy/remove (legacy). Default false: serial stages; Dagger DP is the only concurrent work.
@inline function _seam_nested_threads()
    v = lowercase(strip(get(ENV, "SEAM_CPU_NESTED_THREADS", "0")))
    return v in ("1", "true", "yes", "on")
end

# Optional floor for DP horizontal tile width (wavefront/triangles). Unset: use tile_w as given.
@inline function _effective_dp_tile_w(tile_w::Int, W::Int)::Int
    if !haskey(ENV, "SEAM_MIN_DP_TILE_W")
        return tile_w
    end
    min_w = parse(Int, strip(ENV["SEAM_MIN_DP_TILE_W"]))
    min_w < 1 && error("SEAM_MIN_DP_TILE_W must be >= 1.")
    min_w = min(min_w, W)
    return max(tile_w, min_w)
end

# Max Julia threads used inside one DP row chunk (wavefront / triangles). Default 1 = serial x-loop per chunk.
@inline function _seam_dp_inner_threads()::Int
    if !haskey(ENV, "SEAM_DP_INNER_THREADS")
        return 1
    end
    v = parse(Int, strip(ENV["SEAM_DP_INNER_THREADS"]))
    v < 1 && error("SEAM_DP_INNER_THREADS must be >= 1.")
    return min(v, Threads.nthreads())
end

# Minimum x-span (xr-xl+1) before inner threading splits a row chunk; avoids tiny parallel regions.
@inline function _seam_dp_inner_min_span()::Int
    if !haskey(ENV, "SEAM_DP_INNER_MIN_SPAN")
        return 256
    end
    v = parse(Int, strip(ENV["SEAM_DP_INNER_MIN_SPAN"]))
    v < 1 && error("SEAM_DP_INNER_MIN_SPAN must be >= 1.")
    return v
end

# Minimum x-elements assigned to each inner thread in one row chunk. Prevents over-splitting
# narrow chunks where nested threading overhead dominates useful work.
@inline function _seam_dp_inner_min_chunk()::Int
    if !haskey(ENV, "SEAM_DP_INNER_MIN_CHUNK")
        return 512
    end
    v = parse(Int, strip(ENV["SEAM_DP_INNER_MIN_CHUNK"]))
    v < 1 && error("SEAM_DP_INNER_MIN_CHUNK must be >= 1.")
    return v
end

# If true (default), clamp SEAM_DP_INNER_THREADS by fld(nthreads(), concurrent_outer_tasks) so
# outer Dagger chunks × inner threads does not blow past the process thread pool (avoids descaling).
@inline function _seam_dp_inner_pool_cap()::Bool
    v = lowercase(strip(get(ENV, "SEAM_DP_INNER_POOL_CAP", "1")))
    return !(v in ("0", "false", "no", "off"))
end

@inline function _seam_dp_effective_inner_cap()::Int
    user = _seam_dp_inner_threads()
    _seam_dp_inner_pool_cap() || return user
    outer = max(1, _seam_dp_concurrent_outer_tasks[])
    pool = Threads.nthreads()
    per_outer = max(1, fld(pool, outer))
    return min(user, per_outer)
end

function _seam_wave_size()
    if haskey(ENV, "SEAM_WAVE_SIZE")
        return _parse_wave_size(ENV["SEAM_WAVE_SIZE"])
    end
    # Large ntx: batch more spawns per wave; scale with -t. Override with SEAM_WAVE_SIZE if needed.
    return clamp(4 * Threads.nthreads(), 128, 2048)
end

# When `SEAM_WAVE_SIZE` is unset, never use a wave smaller than the number of tasks in a phase —
# otherwise `_run_task_waves!` serializes multiple fetch batches and hurts scaling (especially at low `-t`).
# When `SEAM_WAVE_SIZE` is set, respect it as a hard cap (no expansion).
@inline function _seam_wave_size_for_task_count(n_tasks::Int)::Int
    w = _seam_wave_size()
    haskey(ENV, "SEAM_WAVE_SIZE") && return w
    return max(w, n_tasks)
end

function _run_task_waves!(builders::Vector{F}, wave_size::Int) where {F<:Function}
    total = length(builders)
    i = 1
    while i <= total
        j = min(i + wave_size - 1, total)
        tasks = Vector{Any}(undef, j - i + 1)
        @inbounds for (k, bi) in enumerate(i:j)
            tasks[k] = builders[bi]()
        end
        foreach(Dagger.fetch, tasks)
        i = j + 1
    end
    return nothing
end

# DP DAG: `ntx = cld(W, tile_w)` column chunks per row wave — at most ~`ntx` concurrent Dagger DP tasks per row
# (each chunk depends on the same/neighbor chunks in the row above). `tile_h` only steps the outer `y0` window;
# it does not merge multiple rows into one Dagger task (still one spawn wave + barrier per row).
# For many threads with modest `ntx`, raise `SEAM_DP_INNER_THREADS` so each chunk parallelizes its x-range (capped).
function cumulative_energy_cpu_wavefront(E::AbstractMatrix{T}; tile_h::Int = 64, tile_w::Int = 64) where T
    H, W = size(E)
    M = Array{T}(undef, H, W)
    B = Array{Int8}(undef, H, W)
    @inbounds begin
        M[1, :] .= E[1, :]
        B[1, :] .= 0
    end
    if H == 1
        return M, B
    end

    strip_h = max(1, tile_h)
    ntx = cld(W, tile_w)
    prev_row_tasks = Any[nothing for _ in 1:ntx]
    wave_size = _seam_wave_size()
    _seam_dp_concurrent_outer_tasks[] = min(ntx, wave_size)

    y0 = 2
    while y0 <= H
        y1 = min(y0 + strip_h - 1, H)
        for y in y0:y1
            row_tasks = Vector{Any}(undef, ntx)
            builders = Vector{Function}(undef, ntx)
            for tx in 1:ntx
                builders[tx] = let tx=tx, y=y, prev_row_tasks=prev_row_tasks
                    () -> begin
                        x1 = (tx - 1) * tile_w + 1
                        x2 = min(tx * tile_w, W)
                        deps = Any[]
                        if prev_row_tasks[tx] !== nothing
                            push!(deps, prev_row_tasks[tx])
                        end
                        if tx > 1 && prev_row_tasks[tx - 1] !== nothing
                            push!(deps, prev_row_tasks[tx - 1])
                        end
                        if tx < ntx && prev_row_tasks[tx + 1] !== nothing
                            push!(deps, prev_row_tasks[tx + 1])
                        end
                        task = Dagger.@spawn dp_row_tile_wait(tuple(deps...), E, M, B, y, x1, x2)
                        row_tasks[tx] = task
                        return task
                    end
                end
            end
            # Wave batching reduces launch bursts while preserving row barrier semantics.
            _run_task_waves!(builders, wave_size)
            prev_row_tasks = row_tasks
        end
        y0 = y1 + 1
    end

    foreach(Dagger.fetch, prev_row_tasks)
    return M, B
end

function cumulative_energy_cpu_wavefront_overlap(img::AbstractMatrix{T}; tile_h::Int = 64, tile_w::Int = 64) where T
    H, W = size(img)
    ET = promote_type(T, Float32)
    E = Array{ET}(undef, H, W)
    M = Array{ET}(undef, H, W)
    B = Array{Int8}(undef, H, W)
    nty = cld(H, tile_h)
    ntx = cld(W, tile_w)
    energy_tasks = Array{Any}(undef, nty, ntx)
    dp_tasks = Array{Any}(undef, nty, ntx)
    for ty in 1:nty, tx in 1:ntx
        y1 = (ty - 1) * tile_h + 1
        y2 = min(ty * tile_h, H)
        x1 = (tx - 1) * tile_w + 1
        x2 = min(tx * tile_w, W)
        energy_tasks[ty, tx] = Dagger.@spawn energy_cpu_tile!(E, img, y1, y2, x1, x2)
    end
    for ty in 1:nty, tx in 1:ntx
        y1 = (ty - 1) * tile_h + 1
        y2 = min(ty * tile_h, H)
        x1 = (tx - 1) * tile_w + 1
        x2 = min(tx * tile_w, W)
        deps = Any[energy_tasks[ty, tx]]
        if ty > 1
            push!(deps, dp_tasks[ty - 1, tx])
            if tx > 1
                push!(deps, dp_tasks[ty - 1, tx - 1])
            end
            if tx < ntx
                push!(deps, dp_tasks[ty - 1, tx + 1])
            end
        end
        dp_tasks[ty, tx] = Dagger.@spawn dp_tile_wait(tuple(deps...), E, M, B, y1, y2, x1, x2)
    end
    for tx in 1:ntx
        Dagger.fetch(dp_tasks[nty, tx])
    end
    return M, B
end

function dp_triangle_down!(E::AbstractMatrix{T}, M::AbstractMatrix{T}, B::AbstractMatrix{Int8},
                           y0::Int, y1::Int, x0::Int, x1::Int) where T
    H, W = size(E)
    @inbounds for y in y0:y1
        r = y - y0
        xl = max(1, x0 - r)
        xr = min(W, x1 + r)
        for x in xl:xr
            left = x > 1 ? M[y - 1, x - 1] : typemax(T)
            mid = M[y - 1, x]
            right = x < W ? M[y - 1, x + 1] : typemax(T)
            if left <= mid && left <= right
                M[y, x] = E[y, x] + left
                B[y, x] = -1
            elseif mid <= right
                M[y, x] = E[y, x] + mid
                B[y, x] = 0
            else
                M[y, x] = E[y, x] + right
                B[y, x] = 1
            end
        end
    end
    return nothing
end

function dp_triangle_up!(E::AbstractMatrix{T}, M::AbstractMatrix{T}, B::AbstractMatrix{Int8},
                         y0::Int, y1::Int, x0::Int, x1::Int) where T
    H, W = size(E)
    @inbounds for y in y0:y1
        r = y - y0
        xl = x0 + r
        xr = x1 - r
        if xl > xr
            break
        end
        xl = max(1, xl)
        xr = min(W, xr)
        for x in xl:xr
            left = x > 1 ? M[y - 1, x - 1] : typemax(T)
            mid = M[y - 1, x]
            right = x < W ? M[y - 1, x + 1] : typemax(T)
            if left <= mid && left <= right
                M[y, x] = E[y, x] + left
                B[y, x] = -1
            elseif mid <= right
                M[y, x] = E[y, x] + mid
                B[y, x] = 0
            else
                M[y, x] = E[y, x] + right
                B[y, x] = 1
            end
        end
    end
    return nothing
end

@inline function _dp_row_range_serial!(E::AbstractMatrix{T}, M::AbstractMatrix{T}, B::AbstractMatrix{Int8},
                                         y::Int, xl::Int, xr::Int, W::Int) where T
    @inbounds for x in xl:xr
        left = x > 1 ? M[y - 1, x - 1] : typemax(T)
        mid = M[y - 1, x]
        right = x < W ? M[y - 1, x + 1] : typemax(T)
        if left <= mid && left <= right
            M[y, x] = E[y, x] + left
            B[y, x] = -1
        elseif mid <= right
            M[y, x] = E[y, x] + mid
            B[y, x] = 0
        else
            M[y, x] = E[y, x] + right
            B[y, x] = 1
        end
    end
    return nothing
end

function dp_row_range!(E::AbstractMatrix{T}, M::AbstractMatrix{T}, B::AbstractMatrix{Int8},
                       y::Int, xl::Int, xr::Int) where T
    W = size(E, 2)
    xl > xr && return nothing
    span = xr - xl + 1
    cap = _seam_dp_effective_inner_cap()
    grain = _seam_dp_inner_min_span()
    if cap <= 1 || span < grain
        _dp_row_range_serial!(E, M, B, y, xl, xr, W)
        return nothing
    end
    min_chunk = _seam_dp_inner_min_chunk()
    # Keep each inner thread on a meaningful contiguous x-range.
    max_threads_by_chunk = max(1, fld(span, min_chunk))
    nt = min(cap, span, max_threads_by_chunk)
    if nt <= 1
        _dp_row_range_serial!(E, M, B, y, xl, xr, W)
        return nothing
    end
    chunk = cld(span, nt)
    # Do not use `@threads :static` here: Julia forbids concurrent/nested :static regions, and many Dagger
    # `dp_row_tile_wait` / `dp_row_range!` tasks can run at once (see threadingconstructs.jl error).
    Threads.@threads for tid in 1:nt
        xlo = xl + (tid - 1) * chunk
        xlo > xr && continue
        xhi = min(xr, xlo + chunk - 1)
        _dp_row_range_serial!(E, M, B, y, xlo, xhi, W)
    end
    return nothing
end

# =============================================================================
# Westrick triangular-blocked strips (MPL `examples/src/seam-carve/SCI.sml`)
#
# Reference: https://shwestrick.github.io/2020/07/29/seam-carve.html
#
# SML uses 0-based row i and column j; Julia uses y = i+1, x = j+1.  `blockWidth` is even;
# `blockHeight = blockWidth ÷ 2`.  `numBlocks = 1 + (W-1) ÷ blockWidth`.
#
# **Upper triangle** `upperTriangle i jMid` (fat top, tip bottom): for k = 0 .. min(H-i, BH)-1
# (half-open range), row i+k, columns j with
#   lo = max(0, jMid - BH + k), hi = min(W, jMid + BH - k),  j ∈ [lo, hi).
#
# **Lower triangle** `lowerTriangle i jMid` (tip top, fat bottom): same k-range, row i+k,
#   lo = max(0, jMid - k - 1), hi = min(W, jMid + k + 1),  j ∈ [lo, hi).
#
# **One strip** at SML start row `i`: (1) all `upperTriangle i (b*BW+BH)` for b = 0..numBlocks-1
# in parallel; (2) all `lowerTriangle (i+1) (b*BW)` for b = 0..numBlocks in parallel.
# Then advance `i += blockHeight + 1`.  This covers BH+1 rows per strip with only **two** barriers.
#
# Row 0 (Julia y=1) is initialized from E like the rest of this module; triangle kernels use the
# same DP recurrence as `_dp_row_range_serial!` for y ≥ 2 and copy E for y = 1.
# =============================================================================

@inline function _seam_westrick_block_width()::Int
    if haskey(ENV, "SEAM_WESTRICK_BLOCK_WIDTH")
        w = parse(Int, strip(ENV["SEAM_WESTRICK_BLOCK_WIDTH"]))
        w % 2 == 0 || error("SEAM_WESTRICK_BLOCK_WIDTH must be even.")
        w < 2 && error("SEAM_WESTRICK_BLOCK_WIDTH must be >= 2.")
        return w
    end
    return 80
end

"""Largest `tile_h` such that `cld(H, tile_h) ≥ T` (for `T > 1`)."""
function _westrick_max_tile_h_for_remove_tasks(H::Int, T::Int)::Int
    T <= 1 && return H
    H >= T || return 0
    return fld(H - 1, T - 1)
end

"""Smallest even `block_width` so upper phase has at least `T` tasks: `1 + (W-1) ÷ bw ≥ T`."""
function _westrick_min_even_block_width_for_upper_tasks(W::Int, T::Int)::Int
    T <= 1 && return 2
    for bw in 2:2:W
        if 1 + (W - 1) ÷ bw >= T
            return bw
        end
    end
    maxb = 1 + (W - 1) ÷ 2
    error("Westrick: width W=$W yields at most $maxb upper-phase tasks; need T=$T (need W ≥ $(2 * T - 1) for bw=2).")
end

"""Largest even `block_width` with `1 + (W-1) ÷ bw ≥ T` (fewest westrick strips / DP barrier pairs for fixed `W`)."""
function _westrick_max_even_block_width_for_upper_tasks(W::Int, T::Int)::Int
    T <= 1 && return min(W, _seam_westrick_block_width())
    best = 2
    for bw in 2:2:W
        if 1 + (W - 1) ÷ bw >= T
            best = bw
        end
    end
    return best
end

@inline function _seam_westrick_block_width_auto_max()::Int
    if haskey(ENV, "SEAM_WESTRICK_BLOCK_WIDTH_AUTO_MAX")
        v = parse(Int, strip(ENV["SEAM_WESTRICK_BLOCK_WIDTH_AUTO_MAX"]))
        v < 2 && error("SEAM_WESTRICK_BLOCK_WIDTH_AUTO_MAX must be >= 2.")
        return v % 2 == 0 ? v : v - 1
    end
    return 512
end

"""
Optional: pick auto `block_width` using this many **notional** threads instead of `Threads.nthreads()` (strong-scaling sweeps keep fixed Westrick strip geometry). When set, `upper_tasks` may be `< Threads.nthreads()` so some workers idle during DP. Combined with `SEAM_WESTRICK_BLOCK_WIDTH_AUTO_TASK_TARGET` as `Tsched = min(S, cap)` when both set.
"""
@inline function _seam_westrick_block_width_sched_threads()::Union{Nothing,Int}
    if !haskey(ENV, "SEAM_WESTRICK_BLOCK_WIDTH_SCHED_THREADS")
        return nothing
    end
    v = parse(Int, strip(ENV["SEAM_WESTRICK_BLOCK_WIDTH_SCHED_THREADS"]))
    v <= 0 && return nothing
    return v
end

"""Optional cap for auto `block_width`: schedule strips as if at most this many upper-phase tasks (fewer strips, wider blocks). When unset, uses full `T` (prior behavior). Some Julia threads may idle in DP when `T` exceeds this cap. Applies to the `T` passed into `_westrick_auto_sched_T` (either `Threads.nthreads()` or `SEAM_WESTRICK_BLOCK_WIDTH_SCHED_THREADS`)."""
@inline function _seam_westrick_block_width_auto_task_target()::Union{Nothing,Int}
    if !haskey(ENV, "SEAM_WESTRICK_BLOCK_WIDTH_AUTO_TASK_TARGET")
        return nothing
    end
    v = parse(Int, strip(ENV["SEAM_WESTRICK_BLOCK_WIDTH_AUTO_TASK_TARGET"]))
    v <= 0 && return nothing
    return v
end

@inline function _westrick_auto_sched_T(T::Int)::Tuple{Int,Int}
    cap = _seam_westrick_block_width_auto_task_target()
    Tsched = cap === nothing ? T : min(T, cap)
    T_coarse = max(Tsched, 2)
    return (Tsched, T_coarse)
end

"""Pick `block_width` for fewer vertical strip iterations. Argument `T` is the thread count used for strip geometry: `Threads.nthreads()` or pinned `SEAM_WESTRICK_BLOCK_WIDTH_SCHED_THREADS`. Enforces `numBlocks ≥ Tsched` for `Tsched = min(T, SEAM_WESTRICK_BLOCK_WIDTH_AUTO_TASK_TARGET)` when the cap is set."""
function _westrick_auto_block_width(W::Int, T::Int)::Int
    # With `Tsched==1`, `_westrick_max_even_block_width_for_upper_tasks(W, 1)` degenerates to default width (~80),
    # leaving ~59 strips on wide images and noisy cold/warm DP times (logs showed `block_width:80` vs `512` at `t=2`).
    # Use at least `T==2` saturation when coarsening strips; enforce `numBlocks ≥ Tsched` below (not necessarily full `T` when task target is set).
    Tsched, T_coarse = _westrick_auto_sched_T(T)
    bw_star = _westrick_max_even_block_width_for_upper_tasks(W, T_coarse)
    capw = _seam_westrick_block_width_auto_max()
    bw = min(bw_star, capw)
    if 1 + (W - 1) ÷ bw < Tsched
        bw = _westrick_min_even_block_width_for_upper_tasks(W, Tsched)
    end
    return bw
end

@inline function _seam_westrick_block_width_auto_enabled()::Bool
    v = lowercase(strip(get(ENV, "SEAM_WESTRICK_BLOCK_WIDTH_AUTO", "0")))
    return v in ("1", "true", "yes", "on")
end

@inline function _resolve_westrick_block_width(W::Int, explicit::Union{Nothing,Int})::Int
    if _seam_westrick_block_width_auto_enabled()
        S = _seam_westrick_block_width_sched_threads()
        T_bw = S === nothing ? Threads.nthreads() : S
        return _westrick_auto_block_width(W, T_bw)
    end
    return something(explicit, _seam_westrick_block_width())
end

"""
    westrick_sync_phase_task_counts(H, W, tile_h, tile_w, block_width) -> NamedTuple

Dagger task counts per **synchronous** phase in [`seam_carve_cpu_dagger_westrick`](@ref) (energy pass; each strip’s upper / lower triangle batch; remove pass):

- `energy` = `cld(H, tile_h) * cld(W, tile_w)`
- `upper` / `lower` = Westrick column blocks for DP (`lower == upper + 1`)
- `remove` = `cld(H, tile_h)`
"""
function westrick_sync_phase_task_counts(H::Int, W::Int, tile_h::Int, tile_w::Int, block_width::Int)
    H >= 1 && W >= 1 || error("H, W must be positive")
    tile_h >= 1 && tile_w >= 1 || error("tile_h, tile_w must be positive")
    block_width % 2 == 0 || error("block_width must be even")
    nty = cld(H, tile_h)
    ntx = cld(W, tile_w)
    numB = 1 + (W - 1) ÷ block_width
    return (; energy = nty * ntx, upper = numB, lower = numB + 1, remove = nty)
end

"""
    westrick_ensure_tasks_per_thread(H, W, T; tile_h, tile_w, block_width)

Return `(tile_h, tile_w, block_width)` tightened so **each** of the Westrick Dagger phases has **at least `T`**
spawned tasks (no nested `Threads.@threads` on this path), when geometry allows:

- **Upper DP:** `numBlocks ≥ T`
- **Lower DP:** `numBlocks + 1 ≥ T` (implied if `numBlocks ≥ T` and `T > 1`)
- **Remove:** `cld(H, tile_h) ≥ T`
- **Energy:** `cld(H, tile_h) * cld(W, tile_w) ≥ T`

Uses the **largest** `tile_h` / `tile_w` / `block_width` that still satisfy those bounds (starting from user values —
only **shrinks** tiles or **shrinks** `block_width` / may replace with a smaller even width when necessary).
Requires `H ≥ T` and roughly `W ≥ 2T - 1` (with `block_width = 2`) so the constraints are feasible.
"""
function westrick_ensure_tasks_per_thread(H::Int, W::Int, T::Int;
        tile_h::Int, tile_w::Int, block_width::Int)
    if T <= 1
        block_width % 2 == 0 || error("block_width must be even")
        return (; tile_h, tile_w, block_width)
    end
    H >= T || error("Westrick: need H ≥ nthreads ($T) so remove phase has ≥T tasks; got H=$H.")
    block_width % 2 == 0 || error("block_width must be even")
    max_upper = 1 + (W - 1) ÷ 2
    max_upper >= T || error("Westrick: width W=$W yields at most $max_upper upper-phase tasks; need T=$T.")

    bw = block_width
    if 1 + (W - 1) ÷ bw < T
        bw = _westrick_min_even_block_width_for_upper_tasks(W, T)
    end

    th_cap = _westrick_max_tile_h_for_remove_tasks(H, T)
    th_cap < 1 && error("Westrick: internal error, remove-task tile_h cap for H=$H, T=$T.")
    th = min(tile_h, th_cap)
    th < 1 && (th = 1)
    nty = cld(H, th)
    if nty < T
        th = 1
        nty = H
        nty >= T || error("Westrick: need H ≥ T for remove phase; H=$H, T=$T.")
    end

    ntx_min = cld(T, nty)
    tw_cap = if ntx_min <= 1
        W
    else
        fld(W - 1, ntx_min - 1)
    end
    tw_cap < 1 && (tw_cap = 1)
    tw = min(tile_w, tw_cap)

    ntx = cld(W, tw)
    if nty * ntx < T
        tw = 1
        ntx = W
        nty * ntx >= T || error("Westrick: cannot reach T=$T energy tasks with H=$H, W=$W (nty=$nty).")
    end

    return (; tile_h = th, tile_w = tw, block_width = bw)
end

@inline function westrick_dp_set_cell!(E::AbstractMatrix{T}, M::AbstractMatrix{T}, B::AbstractMatrix{Int8},
                                       W::Int, y::Int, x::Int) where T
    if y <= 1
        @inbounds begin
            M[1, x] = E[1, x]
            B[1, x] = 0
        end
        return nothing
    end
    @inbounds begin
        left = x > 1 ? M[y - 1, x - 1] : typemax(T)
        mid = M[y - 1, x]
        right = x < W ? M[y - 1, x + 1] : typemax(T)
        if left <= mid && left <= right
            M[y, x] = E[y, x] + left
            B[y, x] = -1
        elseif mid <= right
            M[y, x] = E[y, x] + mid
            B[y, x] = 0
        else
            M[y, x] = E[y, x] + right
            B[y, x] = 1
        end
    end
    return nothing
end

function westrick_upper_triangle!(E::AbstractMatrix{T}, M::AbstractMatrix{T}, B::AbstractMatrix{Int8},
                                  H::Int, W::Int, i_sml::Int, jMid_sml::Int, BH::Int) where T
    klim = min(H - i_sml, BH)
    @inbounds for k in 0:(klim - 1)
        y = i_sml + k + 1
        lo = max(0, jMid_sml - BH + k)
        hi = min(W, jMid_sml + BH - k)
        lo < hi || continue
        for j_sml in lo:(hi - 1)
            x = j_sml + 1
            westrick_dp_set_cell!(E, M, B, W, y, x)
        end
    end
    return nothing
end

function westrick_lower_triangle!(E::AbstractMatrix{T}, M::AbstractMatrix{T}, B::AbstractMatrix{Int8},
                                   H::Int, W::Int, i_sml::Int, jMid_sml::Int, BH::Int) where T
    klim = min(H - i_sml, BH)
    @inbounds for k in 0:(klim - 1)
        y = i_sml + k + 1
        lo = max(0, jMid_sml - k - 1)
        hi = min(W, jMid_sml + k + 1)
        lo < hi || continue
        for j_sml in lo:(hi - 1)
            x = j_sml + 1
            westrick_dp_set_cell!(E, M, B, W, y, x)
        end
    end
    return nothing
end

"""Serial reference for the Westrick strip schedule; for correctness checks only."""
function cumulative_energy_cpu_westrick_serial(E::AbstractMatrix{T}; block_width::Int = _seam_westrick_block_width()) where T
    block_width % 2 == 0 || error("block_width must be even.")
    BH = block_width ÷ 2
    H, W = size(E)
    M = Array{T}(undef, H, W)
    B = Array{Int8}(undef, H, W)
    @inbounds begin
        M[1, :] .= E[1, :]
        B[1, :] .= 0
    end
    H <= 1 && return M, B
    numBlocks = 1 + (W - 1) ÷ block_width
    i_sml = 0
    while i_sml < H
        for b in 0:(numBlocks - 1)
            jMid = b * block_width + BH
            westrick_upper_triangle!(E, M, B, H, W, i_sml, jMid, BH)
        end
        for b in 0:numBlocks
            jMid = b * block_width
            westrick_lower_triangle!(E, M, B, H, W, i_sml + 1, jMid, BH)
        end
        i_sml += BH + 1
    end
    return M, B
end

function cumulative_energy_cpu_westrick(E::AbstractMatrix{T}; block_width::Int = _seam_westrick_block_width(),
                                        wave_size::Int = -1) where T
    block_width % 2 == 0 || error("block_width must be even.")
    BH = block_width ÷ 2
    H, W = size(E)
    M = Array{T}(undef, H, W)
    B = Array{Int8}(undef, H, W)
    @inbounds begin
        M[1, :] .= E[1, :]
        B[1, :] .= 0
    end
    H <= 1 && return M, B
    numBlocks = 1 + (W - 1) ÷ block_width
    i_sml = 0
    while i_sml < H
        builders_up = Function[]
        for b in 0:(numBlocks - 1)
            jMid = b * block_width + BH
            push!(builders_up, let i_sml = i_sml, jMid = jMid
                () -> Dagger.@spawn westrick_upper_triangle!(E, M, B, H, W, i_sml, jMid, BH)
            end)
        end
        nb = length(builders_up)
        w_up = wave_size < 0 ? _seam_wave_size_for_task_count(nb) : wave_size
        _seam_dp_concurrent_outer_tasks[] = nb > 0 ? min(nb, w_up) : 1
        _run_task_waves!(builders_up, w_up)

        builders_lo = Function[]
        for b in 0:numBlocks
            jMid = b * block_width
            push!(builders_lo, let i1 = i_sml + 1, jMid = jMid
                () -> Dagger.@spawn westrick_lower_triangle!(E, M, B, H, W, i1, jMid, BH)
            end)
        end
        nb2 = length(builders_lo)
        w_lo = wave_size < 0 ? _seam_wave_size_for_task_count(nb2) : wave_size
        _seam_dp_concurrent_outer_tasks[] = nb2 > 0 ? min(nb2, w_lo) : 1
        _run_task_waves!(builders_lo, w_lo)

        i_sml += BH + 1
    end
    return M, B
end

function energy_cpu_tile_serial_fill!(E::AbstractMatrix{ET}, img::AbstractMatrix{T},
                                      y1::Int, y2::Int, x1::Int, x2::Int) where {ET,T}
    H, W = size(img)
    @inbounds for y in y1:y2
        ym = y == 1 ? 1 : y - 1
        yp = y == H ? H : y + 1
        for x in x1:x2
            xm = x == 1 ? 1 : x - 1
            xp = x == W ? W : x + 1
            dx = abs(ET(img[y, xp]) - ET(img[y, xm]))
            dy = abs(ET(img[yp, x]) - ET(img[ym, x]))
            E[y, x] = dx + dy
        end
    end
    return nothing
end

function energy_cpu_dagger_tiled_serial(img::AbstractMatrix{T}; tile_h::Int = 256, tile_w::Int = 256) where T
    H, W = size(img)
    ET = promote_type(T, Float32)
    E = Array{ET}(undef, H, W)
    nty = cld(H, tile_h)
    ntx = cld(W, tile_w)
    builders = Function[]
    for ty in 1:nty, tx in 1:ntx
        y1 = (ty - 1) * tile_h + 1
        y2 = min(ty * tile_h, H)
        x1 = (tx - 1) * tile_w + 1
        x2 = min(tx * tile_w, W)
        push!(builders, let y1 = y1, y2 = y2, x1 = x1, x2 = x2
            () -> Dagger.@spawn energy_cpu_tile_serial_fill!(E, img, y1, y2, x1, x2)
        end)
    end
    _seam_dp_concurrent_outer_tasks[] = 1
    nt = length(builders)
    w_e = _seam_wave_size_for_task_count(nt)
    _run_task_waves!(builders, w_e)
    return E
end

function remove_seam_serial_rows!(out::AbstractMatrix{T}, img::AbstractMatrix{T}, seam::Vector{Int},
                                  y1::Int, y2::Int, W::Int) where T
    @inbounds for y in y1:y2
        s = seam[y]
        if s > 1
            copyto!(view(out, y, 1:s - 1), view(img, y, 1:s - 1))
        end
        if s < W
            copyto!(view(out, y, s:W - 1), view(img, y, s + 1:W))
        end
    end
    return nothing
end

function remove_seam_dagger_tiled_serial(img::AbstractMatrix{T}, seam::Vector{Int}; tile_h::Int = 256) where T
    H, W = size(img)
    out = Array{T}(undef, H, W - 1)
    nty = cld(H, tile_h)
    builders = Function[]
    for ty in 1:nty
        y1 = (ty - 1) * tile_h + 1
        y2 = min(ty * tile_h, H)
        push!(builders, let y1 = y1, y2 = y2
            () -> Dagger.@spawn remove_seam_serial_rows!(out, img, seam, y1, y2, W)
        end)
    end
    _seam_dp_concurrent_outer_tasks[] = 1
    nt = length(builders)
    w_r = _seam_wave_size_for_task_count(nt)
    _run_task_waves!(builders, w_r)
    return out
end

function seam_carve_cpu_dagger_westrick(img::AbstractMatrix{T}; k::Int = 1, tile_h::Int = 256, tile_w::Int = 256,
                                        block_width::Union{Nothing,Int} = nothing,
                                        ensure_phase_tasks_ge_threads::Bool = false) where T
    H, W = size(img)
    block_width = _resolve_westrick_block_width(W, block_width)
    if ensure_phase_tasks_ge_threads
        p = westrick_ensure_tasks_per_thread(H, W, Threads.nthreads(); tile_h = tile_h,
            tile_w = tile_w, block_width = block_width)
        tile_h = p.tile_h
        tile_w = p.tile_w
        block_width = p.block_width
    end
    img_cur = img
    dbg = _seam_perf_log_enabled()
    env_snap = dbg ? _perf_env_snapshot() : Dict{String,String}()
    if dbg
        c = westrick_sync_phase_task_counts(H, W, tile_h, tile_w, block_width)
        wdef = _seam_wave_size()
        BH = block_width ÷ 2
        step = BH + 1
        strips = BH > 0 ? cld(H, step) : 0
        ET = promote_type(eltype(img), Float32)
        bpe = sizeof(ET) + sizeof(ET) + sizeof(Int8)
        bytes_dp_arrays = H * W * bpe
        T_bw_dbg = _seam_westrick_block_width_auto_enabled() ?
            something(_seam_westrick_block_width_sched_threads(), Threads.nthreads()) :
            Threads.nthreads()
        Tsched_dbg, T_coarse_dbg = _seam_westrick_block_width_auto_enabled() ? _westrick_auto_sched_T(T_bw_dbg) :
            (Threads.nthreads(), max(Threads.nthreads(), 2))
        d = Dict{String,Any}(
            "runId" => "perf-v2",
            "event" => "westrick_config",
            "bw_auto" => Int(_seam_westrick_block_width_auto_enabled()),
            # Thread count fed into auto block width (pinned S or live nthreads)
            "bw_width_T_input" => T_bw_dbg,
            "bw_sched_threads_env" => something(_seam_westrick_block_width_sched_threads(), 0),
            "bw_auto_Tsched" => Tsched_dbg,
            "bw_auto_T_coarse" => T_coarse_dbg,
            # 0 = env unset / disabled (valid caps are positive)
            "bw_auto_task_target_cap" => something(_seam_westrick_block_width_auto_task_target(), 0),
            "bw_auto_max" => _seam_westrick_block_width_auto_max(),
            "nthreads" => Threads.nthreads(),
            "cpu_threads_sys" => Sys.CPU_THREADS,
            "julia" => string(VERSION),
            "ensure_phase_tasks_ge" => Int(ensure_phase_tasks_ge_threads),
            "H" => H, "W" => W, "tile_h" => tile_h, "tile_w" => tile_w,
            "nty_energy" => cld(H, tile_h),
            "ntx_energy" => cld(W, tile_w),
            "block_width" => block_width,
            "BH" => BH,
            "strip_step_rows" => step,
            "energy_tasks" => c.energy,
            "upper_tasks" => c.upper, "lower_tasks" => c.lower, "remove_tasks" => c.remove,
            "westrick_strips_est" => strips,
            "dp_global_barrier_pairs_est" => 2 * strips,
            "upper_tasks_per_strip" => c.upper,
            "wave_size_base" => wdef,
            "wave_size_upper_cap" => _seam_wave_size_for_task_count(c.upper),
            "wave_size_lower_cap" => _seam_wave_size_for_task_count(c.lower),
            "waves_upper_per_strip" => cld(c.upper, _seam_wave_size_for_task_count(c.upper)),
            "waves_lower_per_strip" => cld(c.lower, _seam_wave_size_for_task_count(c.lower)),
            "bytes_est_img" => sizeof(img_cur),
            "bytes_est_E_M_B" => bytes_dp_arrays,
            "env_keys_set" => length(env_snap),
        )
        merge!(d, Dict{String,Any}("ENV_" * k => v for (k, v) in env_snap))
        _agent_debug_log(;
            hypothesisId = "H1",
            location = "DaggerSeamCarving.jl:seam_carve_cpu_dagger_westrick:config",
            message = "westrick perf snapshot + schedule geometry",
            data = d,
        )
    end
    for kpass in 1:k
        local te, td, tr, tfind
        local be, bd, br, gce, gcd, gcr, gcf
        if dbg
            st_e = @timed energy_cpu_dagger_tiled_serial(img_cur; tile_h = tile_h, tile_w = tile_w)
            E = st_e.value
            te = st_e.time
            be = st_e.bytes
            gce = st_e.gctime
            st_d = @timed cumulative_energy_cpu_westrick(E; block_width = block_width)
            M, B = st_d.value
            td = st_d.time
            bd = st_d.bytes
            gcd = st_d.gctime
            st_f = @timed find_seam(M, B)
            seam = st_f.value
            tfind = st_f.time
            gcf = st_f.gctime
            st_r = @timed remove_seam_dagger_tiled_serial(img_cur, seam; tile_h = tile_h)
            img_cur = st_r.value
            tr = st_r.time
            br = st_r.bytes
            gcr = st_r.gctime
            tot = te + td + tfind + tr
            d2 = Dict{String,Any}(
                "runId" => "perf-v2",
                "event" => "westrick_timings",
                "k_pass" => kpass,
                "k_total" => k,
                "nthreads" => Threads.nthreads(),
                "energy_s" => te,
                "westrick_dp_s" => td,
                "find_seam_s" => tfind,
                "remove_s" => tr,
                "total_s" => tot,
                "dp_frac" => (tot > 0 ? td / tot : 0.0),
                "alloc_bytes_energy" => be,
                "alloc_bytes_westrick_dp" => bd,
                "alloc_bytes_remove" => br,
                "gc_s_energy" => gce,
                "gc_s_dp" => gcd,
                "gc_s_find" => gcf,
                "gc_s_remove" => gcr,
                "throughput_M_cell_s" => (tot > 0 ? (H * W) / (tot * 1.0e6) : 0.0),
            )
            _agent_debug_log(;
                hypothesisId = "H2",
                location = "DaggerSeamCarving.jl:seam_carve_cpu_dagger_westrick:timings",
                message = "per-phase time, allocations, gc (one k pass)",
                data = d2,
            )
        else
            E = energy_cpu_dagger_tiled_serial(img_cur; tile_h = tile_h, tile_w = tile_w)
            M, B = cumulative_energy_cpu_westrick(E; block_width = block_width)
            seam = find_seam(M, B)
            img_cur = remove_seam_dagger_tiled_serial(img_cur, seam; tile_h = tile_h)
        end
    end
    return img_cur
end

"""
    westrick_dp_matches_serial(E; kwargs...) -> Bool

Compare `cumulative_energy_cpu_westrick_serial` and `cumulative_energy_cpu_serial` on `E`.
"""
function westrick_dp_matches_serial(E::AbstractMatrix; kwargs...)
    M1, B1 = cumulative_energy_cpu_serial(E)
    M2, B2 = cumulative_energy_cpu_westrick_serial(E; kwargs...)
    return M1 == M2 && B1 == B2
end

# Same as `westrick_dp_matches_serial` but exercises the Dagger DP path.
function westrick_dagger_dp_matches_serial(E::AbstractMatrix; kwargs...)
    M1, B1 = cumulative_energy_cpu_serial(E)
    M2, B2 = cumulative_energy_cpu_westrick(E; kwargs...)
    return M1 == M2 && B1 == B2
end

# Same DP row barriers as wavefront; row slices call `dp_row_range!` (honors `SEAM_DP_INNER_THREADS`).
function cumulative_energy_cpu_triangles(E::AbstractMatrix{T}; tile_h::Int = 64, tile_w::Int = 128) where T
    H, W = size(E)
    M = Array{T}(undef, H, W)
    B = Array{Int8}(undef, H, W)
    @inbounds begin
        M[1, :] .= E[1, :]
        B[1, :] .= 0
    end
    if H == 1
        return M, B
    end

    strip_h = max(1, tile_h)
    base_w = max(tile_w, 2 * strip_h)
    nseg = cld(W, base_w)
    covered = Vector{Bool}(undef, W)
    wave_size = _seam_wave_size()

    y0 = 2
    while y0 <= H
        y1 = min(y0 + strip_h - 1, H)
        # Triangle-strip schedule:
        # 1) odd segments form "down" triangle windows
        # 2) even segments form "up" triangle windows
        # We enforce a row barrier so every cell in row y is finalized before y+1,
        # matching DP dependencies on M[y-1, x-1:x+1].
        for y in y0:y1
            r = y - y0
            fill!(covered, false)
            builders = Function[]

            # Odd segments: expanding "down" triangles.
            for seg in 1:nseg
                if isodd(seg)
                    x0 = (seg - 1) * base_w + 1
                    x1 = min(seg * base_w, W)
                    xl = max(1, x0 - r)
                    xr = min(W, x1 + r)
                    if xl <= xr
                        @inbounds covered[xl:xr] .= true
                        push!(builders, let y=y, xl=xl, xr=xr
                            () -> Dagger.@spawn dp_row_range!(E, M, B, y, xl, xr)
                        end)
                    end
                end
            end

            # Even segments: shrinking "up" triangles.
            for seg in 2:2:nseg
                x0 = (seg - 1) * base_w + 1
                x1 = min(seg * base_w, W)
                xl = max(1, x0 + r)
                xr = min(W, x1 - r)
                if xl <= xr
                    @inbounds covered[xl:xr] .= true
                    push!(builders, let y=y, xl=xl, xr=xr
                        () -> Dagger.@spawn dp_row_range!(E, M, B, y, xl, xr)
                    end)
                end
            end

            # Ensure full row coverage for correctness if triangle windows leave gaps.
            x = 1
            while x <= W
                while x <= W && covered[x]
                    x += 1
                end
                x > W && break
                xl = x
                while x <= W && !covered[x]
                    x += 1
                end
                xr = x - 1
                push!(builders, let y=y, xl=xl, xr=xr
                    () -> Dagger.@spawn dp_row_range!(E, M, B, y, xl, xr)
                end)
            end

            # Row barrier: DP row y must be complete before y+1 starts.
            nb = length(builders)
            _seam_dp_concurrent_outer_tasks[] = nb > 0 ? min(nb, wave_size) : 1
            _run_task_waves!(builders, wave_size)
        end
        y0 = y1 + 1
    end

    return M, B
end

function find_seam(M::AbstractMatrix{T}, B::AbstractMatrix{Int8}) where T
    H, W = size(M)
    seam = Vector{Int}(undef, H)
    minv = M[H, 1]
    idx = 1
    @inbounds for x in 2:W
        if M[H, x] < minv
            minv = M[H, x]
            idx = x
        end
    end
    seam[H] = idx
    @inbounds for y in (H - 1):-1:1
        seam[y] = clampi(seam[y + 1] + B[y + 1, seam[y + 1]], 1, W)
    end
    return seam
end

@inline function find_seam_from_mb(MB)
    M, B = MB
    return find_seam(M, B)
end

function remove_seam(img::AbstractMatrix{T}, seam::Vector{Int}) where T
    H, W = size(img)
    out = Array{T}(undef, H, W - 1)
    Threads.@threads for y in 1:H
        s = seam[y]
        if s > 1
            @inbounds copyto!(view(out, y, 1:s - 1), view(img, y, 1:s - 1))
        end
        if s < W
            @inbounds copyto!(view(out, y, s:W - 1), view(img, y, s + 1:W))
        end
    end
    return out
end

function seam_carve_cpu_dagger(img::AbstractMatrix{T}; k::Int = 1) where T
    img_cur = img
    for _ in 1:k
        E = Dagger.@spawn energy_cpu(img_cur)
        MB = Dagger.@spawn cumulative_energy_cpu(E)
        seam = Dagger.@spawn find_seam_from_mb(MB)
        img_cur = Dagger.@spawn remove_seam(img_cur, seam)
    end
    return Dagger.fetch(img_cur)
end


function energy_cpu_tile(img::AbstractMatrix{T}, y1::Int, y2::Int, x1::Int, x2::Int) where T
    H, W = size(img)
    ET = promote_type(T, Float32)
    tile = Array{ET}(undef, y2 - y1 + 1, x2 - x1 + 1)
    Threads.@threads for y in y1:y2
        ym = y == 1 ? 1 : y - 1
        yp = y == H ? H : y + 1
        @inbounds for x in x1:x2
            xm = x == 1 ? 1 : x - 1
            xp = x == W ? W : x + 1
            dx = abs(ET(img[y, xp]) - ET(img[y, xm]))
            dy = abs(ET(img[yp, x]) - ET(img[ym, x]))
            tile[y - y1 + 1, x - x1 + 1] = dx + dy
        end
    end
    return tile
end

function energy_cpu_tile!(E::AbstractMatrix{ET}, img::AbstractMatrix{T}, y1::Int, y2::Int, x1::Int, x2::Int) where {ET,T}
    H, W = size(img)
    Threads.@threads for y in y1:y2
        ym = y == 1 ? 1 : y - 1
        yp = y == H ? H : y + 1
        @inbounds for x in x1:x2
            xm = x == 1 ? 1 : x - 1
            xp = x == W ? W : x + 1
            dx = abs(ET(img[y, xp]) - ET(img[y, xm]))
            dy = abs(ET(img[yp, x]) - ET(img[ym, x]))
            E[y, x] = dx + dy
        end
    end
    return nothing
end

function energy_cpu_dagger_tiled(img::AbstractMatrix{T}; tile_h::Int = 256, tile_w::Int = 256) where T
    H, W = size(img)
    nty = cld(H, tile_h)
    ntx = cld(W, tile_w)
    tasks = Array{Any}(undef, nty, ntx)
    for ty in 1:nty, tx in 1:ntx
        y1 = (ty - 1) * tile_h + 1
        y2 = min(ty * tile_h, H)
        x1 = (tx - 1) * tile_w + 1
        x2 = min(tx * tile_w, W)
        tasks[ty, tx] = Dagger.@spawn energy_cpu_tile(img, y1, y2, x1, x2)
    end
    E = Array{promote_type(T, Float32)}(undef, H, W)
    for ty in 1:nty, tx in 1:ntx
        y1 = (ty - 1) * tile_h + 1
        y2 = min(ty * tile_h, H)
        x1 = (tx - 1) * tile_w + 1
        x2 = min(tx * tile_w, W)
        tile = Dagger.fetch(tasks[ty, tx])
        @inbounds E[y1:y2, x1:x2] .= tile
    end
    return E
end

function remove_seam_block(img::AbstractMatrix{T}, seam::Vector{Int}, y1::Int, y2::Int) where T
    H, W = size(img)
    block = Array{T}(undef, y2 - y1 + 1, W - 1)
    Threads.@threads for y in y1:y2
        s = seam[y]
        if s > 1
            @inbounds copyto!(view(block, y - y1 + 1, 1:s - 1), view(img, y, 1:s - 1))
        end
        if s < W
            @inbounds copyto!(view(block, y - y1 + 1, s:W - 1), view(img, y, s + 1:W))
        end
    end
    return block
end

function remove_seam_dagger_tiled(img::AbstractMatrix{T}, seam::Vector{Int}; tile_h::Int = 256) where T
    H, W = size(img)
    nty = cld(H, tile_h)
    tasks = Vector{Any}(undef, nty)
    for ty in 1:nty
        y1 = (ty - 1) * tile_h + 1
        y2 = min(ty * tile_h, H)
        tasks[ty] = Dagger.@spawn remove_seam_block(img, seam, y1, y2)
    end
    out = Array{T}(undef, H, W - 1)
    for ty in 1:nty
        y1 = (ty - 1) * tile_h + 1
        y2 = min(ty * tile_h, H)
        block = Dagger.fetch(tasks[ty])
        @inbounds out[y1:y2, :] .= block
    end
    return out
end

function seam_carve_cpu_dagger_tiled(img::AbstractMatrix{T}; k::Int = 1, tile_h::Int = 256, tile_w::Int = 256) where T
    img_cur = img
    for _ in 1:k
        E = energy_cpu_dagger_tiled(img_cur; tile_h=tile_h, tile_w=tile_w)
        M, B = cumulative_energy_cpu(E)
        seam = find_seam(M, B)
        img_cur = remove_seam_dagger_tiled(img_cur, seam; tile_h=tile_h)
    end
    return img_cur
end


function seam_carve_cpu_dagger_wavefront(img::AbstractMatrix{T}; k::Int = 1, tile_h::Int = 64, tile_w::Int = 64) where T
    img_cur = img
    nested = _seam_nested_threads()
    for _ in 1:k
        E = nested ? energy_cpu(img_cur) : energy_cpu_serial(img_cur)
        tw = _effective_dp_tile_w(tile_w, size(img_cur, 2))
        M, B = cumulative_energy_cpu_wavefront(E; tile_h=tile_h, tile_w=tw)
        seam = find_seam(M, B)
        img_cur = nested ? remove_seam(img_cur, seam) : remove_seam_serial(img_cur, seam)
    end
    return img_cur
end

function seam_step_tileoverlap(img::AbstractMatrix{T}; tile_h::Int = 64, tile_w::Int = 64) where T
    M, B = cumulative_energy_cpu_wavefront_overlap(img; tile_h=tile_h, tile_w=tile_w)
    seam = find_seam(M, B)
    return remove_seam(img, seam)
end

function seam_carve_cpu_dagger_tileoverlap(img::AbstractMatrix{T}; k::Int = 1, tile_h::Int = 64, tile_w::Int = 64) where T
    img_cur = img
    for _ in 1:k
        img_cur = seam_step_tileoverlap(img_cur; tile_h=tile_h, tile_w=tile_w)
    end
    return img_cur
end


function seam_carve_cpu_dagger_triangles(img::AbstractMatrix{T}; k::Int = 1, tile_h::Int = 64, tile_w::Int = 128) where T
    img_cur = img
    nested = _seam_nested_threads()
    for _ in 1:k
        E = nested ? energy_cpu(img_cur) : energy_cpu_serial(img_cur)
        tw = _effective_dp_tile_w(tile_w, size(img_cur, 2))
        M, B = cumulative_energy_cpu_triangles(E; tile_h=tile_h, tile_w=tw)
        seam = find_seam(M, B)
        img_cur = nested ? remove_seam(img_cur, seam) : remove_seam_serial(img_cur, seam)
    end
    return img_cur
end

function energy_cpu_serial(img::AbstractMatrix{T}) where T
    H, W = size(img)
    ET = promote_type(T, Float32)
    E = Array{ET}(undef, H, W)
    for y in 1:H
        ym = y == 1 ? 1 : y - 1
        yp = y == H ? H : y + 1
        @inbounds for x in 1:W
            xm = x == 1 ? 1 : x - 1
            xp = x == W ? W : x + 1
            dx = abs(ET(img[y, xp]) - ET(img[y, xm]))
            dy = abs(ET(img[yp, x]) - ET(img[ym, x]))
            E[y, x] = dx + dy
        end
    end
    return E
end

function cumulative_energy_cpu_serial(E::AbstractMatrix{T}) where T
    H, W = size(E)
    M = Array{T}(undef, H, W)
    B = Array{Int8}(undef, H, W)
    @inbounds begin
        M[1, :] .= E[1, :]
        B[1, :] .= 0
    end
    for y in 2:H
        @inbounds for x in 1:W
            left = x > 1 ? M[y - 1, x - 1] : typemax(T)
            mid = M[y - 1, x]
            right = x < W ? M[y - 1, x + 1] : typemax(T)
            if left <= mid && left <= right
                M[y, x] = E[y, x] + left
                B[y, x] = -1
            elseif mid <= right
                M[y, x] = E[y, x] + mid
                B[y, x] = 0
            else
                M[y, x] = E[y, x] + right
                B[y, x] = 1
            end
        end
    end
    return M, B
end

function remove_seam_serial(img::AbstractMatrix{T}, seam::Vector{Int}) where T
    H, W = size(img)
    out = Array{T}(undef, H, W - 1)
    for y in 1:H
        s = seam[y]
        if s > 1
            @inbounds copyto!(view(out, y, 1:s - 1), view(img, y, 1:s - 1))
        end
        if s < W
            @inbounds copyto!(view(out, y, s:W - 1), view(img, y, s + 1:W))
        end
    end
    return out
end

function seam_carve_cpu_serial(img::AbstractMatrix{T}; k::Int = 1) where T
    img_cur = img
    for _ in 1:k
        E = energy_cpu_serial(img_cur)
        M, B = cumulative_energy_cpu_serial(E)
        seam = find_seam(M, B)
        img_cur = remove_seam_serial(img_cur, seam)
    end
    return img_cur
end


const KA = KernelAbstractions

@kernel function energy_kernel!(E, I, H::Int, W::Int)
    idx = @index(Global)
    if idx <= H * W
        y = (idx - 1) ÷ W + 1
        x = (idx - 1) % W + 1
        xm = x == 1 ? 1 : x - 1
        xp = x == W ? W : x + 1
        ym = y == 1 ? 1 : y - 1
        yp = y == H ? H : y + 1
        E[y, x] = abs(I[y, xp] - I[y, xm]) + abs(I[yp, x] - I[ym, x])
    end
end

function energy_gpu(img::AbstractArray{T, 2}) where T
    H, W = size(img)
    E = similar(img)
    backend = KA.get_backend(img)
    threads = DEFAULT_THREADS
    energy_kernel!(backend, threads)(E, img, H, W; ndrange=H * W)
    KA.synchronize(backend)
    return E
end

@kernel function dp_row_kernel!(M, B, E, y::Int, W::Int)
    x = @index(Global)
    if x <= W
        left = x > 1 ? M[y - 1, x - 1] : typemax(eltype(M))
        mid = M[y - 1, x]
        right = x < W ? M[y - 1, x + 1] : typemax(eltype(M))
        if left <= mid && left <= right
            M[y, x] = E[y, x] + left
            B[y, x] = Int8(-1)
        elseif mid <= right
            M[y, x] = E[y, x] + mid
            B[y, x] = Int8(0)
        else
            M[y, x] = E[y, x] + right
            B[y, x] = Int8(1)
        end
    end
end

function cumulative_energy_gpu(E::AbstractArray{T, 2}) where T
    H, W = size(E)
    M = similar(E)
    B = similar(E, Int8, H, W)
    M[1, :] .= E[1, :]
    B[1, :] .= 0
    backend = KA.get_backend(E)
    threads = DEFAULT_THREADS
    for y in 2:H
        dp_row_kernel!(backend, threads)(M, B, E, y, W; ndrange=W)
    end
    KA.synchronize(backend)
    return M, B
end

@kernel function seam_backtrack_kernel!(seam, M, B, H::Int, W::Int)
    i = @index(Global)
    if i == 1
        minv = M[H, 1]
        idx = 1
        for x in 2:W
            v = M[H, x]
            if v < minv
                minv = v
                idx = x
            end
        end
        seam[H] = idx
        for y in (H - 1):-1:1
            seam[y] = clampi(seam[y + 1] + B[y + 1, seam[y + 1]], 1, W)
        end
    end
end

function find_seam_gpu_device(M::AbstractArray{T, 2}, B::AbstractArray{Int8, 2}) where T
    H, W = size(M)
    seam = similar(M, Int32, H)
    backend = KA.get_backend(M)
    seam_backtrack_kernel!(backend, 1)(seam, M, B, H, W; ndrange=1)
    KA.synchronize(backend)
    return seam
end

function find_seam_gpu(M::AbstractArray{T, 2}, B::AbstractArray{Int8, 2}) where T
    H, W = size(M)
    last = Array(view(M, H, :))
    minv = last[1]
    idx = 1
    @inbounds for x in 2:W
        if last[x] < minv
            minv = last[x]
            idx = x
        end
    end
    B_cpu = Array(B)
    seam = Vector{Int}(undef, H)
    seam[H] = idx
    @inbounds for y in (H - 1):-1:1
        seam[y] = clampi(seam[y + 1] + B_cpu[y + 1, seam[y + 1]], 1, W)
    end
    return seam
end

@inline function find_seam_gpu_from_mb(MB)
    M, B = MB
    return find_seam_gpu(M, B)
end

@kernel function remove_seam_kernel!(out, img, seam, H::Int, W::Int)
    idx = @index(Global)
    if idx <= H * (W - 1)
        y = (idx - 1) ÷ (W - 1) + 1
        x = (idx - 1) % (W - 1) + 1
        s = seam[y]
        srcx = x < s ? x : x + 1
        out[y, x] = img[y, srcx]
    end
end

function remove_seam_gpu(img::AbstractArray{T, 2}, seam::Vector{Int}) where T
    H, W = size(img)
    out = similar(img, T, H, W - 1)
    seam_d = similar(img, Int32, H)
    copyto!(seam_d, Int32.(seam))
    backend = KA.get_backend(img)
    threads = DEFAULT_THREADS
    remove_seam_kernel!(backend, threads)(out, img, seam_d, H, W; ndrange=H * (W - 1))
    KA.synchronize(backend)
    return out
end

function remove_seam_gpu_device(img::AbstractArray{T, 2}, seam_d::AbstractArray{Int32}) where T
    H, W = size(img)
    out = similar(img, T, H, W - 1)
    backend = KA.get_backend(img)
    threads = DEFAULT_THREADS
    remove_seam_kernel!(backend, threads)(out, img, seam_d, H, W; ndrange=H * (W - 1))
    KA.synchronize(backend)
    return out
end

function seam_carve_gpu_dagger(img::AbstractArray{T, 2}; k::Int = 1) where T
    img_cur = img
    for _ in 1:k
        E = Dagger.@spawn energy_gpu(img_cur)
        MB = Dagger.@spawn cumulative_energy_gpu(E)
        seam = Dagger.@spawn find_seam_gpu_from_mb(MB)
        img_cur = Dagger.@spawn remove_seam_gpu(img_cur, seam)
    end
    return Dagger.fetch(img_cur)
end

function seam_carve_gpu_dagger_device(img::AbstractArray{T, 2}; k::Int = 1) where T
    img_cur = img
    for _ in 1:k
        E = Dagger.@spawn energy_gpu(img_cur)
        MB = Dagger.@spawn cumulative_energy_gpu(E)
        seam_d = Dagger.@spawn find_seam_gpu_device_from_mb(MB)
        img_cur = Dagger.@spawn remove_seam_gpu_device(img_cur, seam_d)
    end
    return Dagger.fetch(img_cur)
end

@inline function find_seam_gpu_device_from_mb(MB)
    M, B = MB
    return find_seam_gpu_device(M, B)
end

end
