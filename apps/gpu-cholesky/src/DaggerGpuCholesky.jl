"""
GPU-backed DArray Cholesky benchmark helpers: four-GPU 2×2 block-cyclic assignment,
SPD matrix as `ones` plus diagonal boost, and optional correctness checks.

Load **one** GPU package (`CUDA`, `AMDGPU`, `oneAPI`, or `Metal`) in `Main` **before** `Dagger`;
this module does not import a vendor stack at load time.
"""
module DaggerGpuCholesky

using Dagger
using GPUArrays: unsafe_free!
using GPUArraysCore: AbstractGPUArray, allowscalar
using LinearAlgebra

export DEVICE_BACKENDS,
    CHOLESKY_ALGORITHMS,
    resolve_device,
    resolve_device_strict,
    device_from_loaded,
    four_gpu_processors,
    cholesky_block_cyclic_assignment,
    spd_ones_darray,
    spd_ones_dense_vendor,
    wait_darray!,
    sync_darray!,
    bench_cholesky_once!,
    bench_cholesky_inplace_once!,
    bench_vendor_cholesky_once!,
    verify_cholesky_small,
    purge_gpu_memory!,
    unsafe_free_darray!,
    vendor_timing_sync!,
    vendor_gpu_reclaim!,
    unsafe_free_dense_gpu!

"""
    CHOLESKY_ALGORITHMS

Supported algorithm symbols for `bench_cholesky_once!(DA, algo)`:

- `:rl`    – right-looking (original Dagger, no pinning / lookahead)
- `:rl_la` – right-looking + processor pinning + lookahead (recommended)
- `:ll`    – left-looking + processor pinning (GEMM-based updates)
"""
const CHOLESKY_ALGORITHMS = (:rl, :rl_la, :ll)

const DEVICE_BACKENDS = (
    (:cuda, :CUDA, :CuArray),
    (:amdgpu, :AMDGPU, :ROCArray),
    (:oneapi, :oneAPI, :oneArray),
    (:metal, :Metal, :MtlArray),
)

const GPU_PROC_NAMES = (
    :CuArrayDeviceProc,
    :ROCArrayDeviceProc,
    :oneArrayDeviceProc,
    :MtlArrayDeviceProc,
)

function _is_gpu_array_processor(p::Dagger.Processor)
    return nameof(typeof(p)) in GPU_PROC_NAMES
end

function _device_sort_key(p::Dagger.Processor)
    Tp = typeof(p)
    id = if hasfield(Tp, :device)
        Int(getfield(p, :device)::Integer)
    elseif hasfield(Tp, :device_id)
        Int(getfield(p, :device_id)::Integer)
    else
        0
    end
    return (id, string(nameof(Tp)))
end

"""
    resolve_device() -> Symbol

Read `CHOLESKY_DEVICE` env: `auto`, `cuda`, `amdgpu`, `oneapi`, `metal`.
"""
function resolve_device()
    raw = lowercase(strip(get(ENV, "CHOLESKY_DEVICE", "auto")))
    if raw in ("", "auto")
        return :auto
    elseif raw in ("cuda", "nvidia")
        return :cuda
    elseif raw in ("amdgpu", "rocm", "amd")
        return :amdgpu
    elseif raw in ("oneapi", "intel")
        return :oneapi
    elseif raw in ("metal", "apple")
        return :metal
    else
        error("Unknown CHOLESKY_DEVICE: $raw. Use auto|cuda|amdgpu|oneapi|metal.")
    end
end

function device_from_loaded()
    for (device, modsym, ctor) in DEVICE_BACKENDS
        if isdefined(Main, modsym)
            mod = getfield(Main, modsym)
            if isdefined(mod, ctor)
                return device
            end
        end
    end
    error(
        "No GPU backend loaded. Start Julia with e.g. `using CUDA` (or AMDGPU, oneAPI, Metal) " *
        "before `using Dagger` (see Dagger.jl docs), then load this module.",
    )
end

function resolve_device_strict()
    d = resolve_device()
    d === :auto && return device_from_loaded()
    # Ensure requested backend is loaded
    for (key, modsym, ctor) in DEVICE_BACKENDS
        key === d || continue
        isdefined(Main, modsym) || error("CHOLESKY_DEVICE=$d but Main.$modsym is not defined; load it first.")
        mod = getfield(Main, modsym)
        isdefined(mod, ctor) || error("Module $modsym does not define $ctor.")
        return d
    end
    error("Unreachable")
end

"""
    four_gpu_processors() -> Vector{<:Dagger.Processor}

Return four distinct GPU array processors for the current OS process, sorted by device id.
Requires ≥4 visible devices of one backend.
"""
function four_gpu_processors()
    os = Dagger.OSProc()
    allp = collect(Dagger.get_processors(os))
    gpus = filter(_is_gpu_array_processor, allp)
    isempty(gpus) && error("No GPU Dagger processors found. Load one of CUDA, AMDGPU, oneAPI, or Metal before Dagger.")
    sort!(gpus; by=_device_sort_key)
    length(gpus) < 4 &&
        error("Need at least 4 GPU devices for this benchmark; found $(length(gpus)).")
    return gpus[1:4]
end

"""
    cholesky_block_cyclic_assignment(gpu_procs::AbstractVector{<:Dagger.Processor}, n_blocks::Int)

2×2 block-cyclic processor grid (ScaLAPACK-style): block `(i,j)` → `gpu_procs[mod(i-1,2)+2*mod(j-1,2)+1]`.
"""
function cholesky_block_cyclic_assignment(
    gpu_procs::AbstractVector{P},
    n_blocks::Int,
) where {P<:Dagger.Processor}
    length(gpu_procs) == 4 || throw(ArgumentError("need exactly 4 processors"))
    n_blocks ≥ 1 || throw(ArgumentError("n_blocks must be ≥ 1"))
    assignment = Matrix{P}(undef, n_blocks, n_blocks)
    for j in 1:n_blocks, i in 1:n_blocks
        idx = mod(i - 1, 2) + 2 * mod(j - 1, 2) + 1
        assignment[i, j] = gpu_procs[idx]
    end
    return assignment
end

# Host scalar diagonal loop is invalid on GPU arrays without allowscalar (GPUArrays).
function _diag_boost!(A::AbstractGPUArray{T,2}, δ::T) where {T}
    n = min(size(A, 1), size(A, 2))
    allowscalar() do
        @inbounds for i in 1:n
            A[i, i] += δ
        end
    end
    return A
end

function _diag_boost!(A::AbstractMatrix{T}, δ::T) where {T}
    n = min(size(A, 1), size(A, 2))
    @inbounds for i in 1:n
        A[i, i] += δ
    end
    return A
end

function _vendor_select_device!(dev::Symbol, device_id::Int)
    device_id ≥ 0 || throw(ArgumentError("device_id must be ≥ 0 (CHOLESKY_VENDOR_DEVICE)"))
    if dev === :cuda
        devs = Main.CUDA.devices()
        device_id < length(devs) ||
            throw(ArgumentError("CHOLESKY_VENDOR_DEVICE=$device_id out of range ($(length(devs)) CUDA devices)"))
        Main.CUDA.device!(Main.CUDA.CuDevice(device_id))
    elseif dev === :amdgpu
        isdefined(Main, :AMDGPU) || error("AMDGPU not loaded in Main")
        M = Main.AMDGPU
        ds = collect(M.devices())
        device_id < length(ds) ||
            throw(ArgumentError("CHOLESKY_VENDOR_DEVICE=$device_id out of range ($(length(ds)) AMDGPU devices)"))
        M.device!(ds[device_id + 1])
    elseif dev === :oneapi
        isdefined(Main, :oneAPI) || error("oneAPI not loaded in Main")
        O = Main.oneAPI
        ds = collect(O.devices())
        device_id < length(ds) ||
            throw(ArgumentError("CHOLESKY_VENDOR_DEVICE=$device_id out of range ($(length(ds)) oneAPI devices)"))
        O.device!(ds[device_id + 1])
    elseif dev === :metal
        isdefined(Main, :Metal) || error("Metal not loaded in Main")
        Mt = Main.Metal
        ds = Mt.devices()
        device_id < length(ds) ||
            throw(ArgumentError("CHOLESKY_VENDOR_DEVICE=$device_id out of range ($(length(ds)) Metal devices)"))
        Mt.device!(ds[device_id + 1])
    else
        error("Unknown device symbol: $dev")
    end
    return nothing
end

function _vendor_dense_ones(::Type{T}, N::Int, dev::Symbol, device_id::Int) where {T<:AbstractFloat}
    _vendor_select_device!(dev, device_id)
    if dev === :cuda
        return Main.CUDA.fill(one(T), N, N)
    elseif dev === :amdgpu
        M = Main.AMDGPU
        A = M.ROCArray{T}(undef, N, N)
        fill!(A, one(T))
        return A
    elseif dev === :oneapi
        O = Main.oneAPI
        A = O.oneArray{T}(undef, N, N)
        A .= one(T)
        return A
    elseif dev === :metal
        Mt = Main.Metal
        A = Mt.MtlArray{T}(undef, N, N)
        fill!(A, one(T))
        return A
    end
    error("Cannot allocate dense GPU matrix for backend: $dev")
end

function _vendor_synchronize(A::AbstractArray)
    if isdefined(Main, :CUDA) && A isa Main.CUDA.CuArray
        Main.CUDA.synchronize()
    elseif isdefined(Main, :AMDGPU) && A isa Main.AMDGPU.ROCArray
        Main.AMDGPU.synchronize()
    elseif isdefined(Main, :oneAPI) && A isa Main.oneAPI.oneArray
        Main.oneAPI.synchronize()
    elseif isdefined(Main, :Metal) && A isa Main.Metal.MtlArray
        Main.Metal.synchronize()
    end
    return nothing
end

function _sync_all_gpu_devices!(dev::Symbol)
    if dev === :cuda
        C = Main.CUDA
        cur = C.device()
        for d in C.devices()
            C.device!(d)
            C.synchronize()
        end
        C.device!(cur)
    elseif dev === :amdgpu
        M = Main.AMDGPU
        cur = M.device()
        for d in M.devices()
            M.device!(d)
            M.synchronize()
        end
        M.device!(cur)
    elseif dev === :oneapi
        O = Main.oneAPI
        cur = O.device()
        for d in O.devices()
            O.device!(d)
            O.synchronize()
        end
        O.device!(cur)
    elseif dev === :metal
        Mt = Main.Metal
        for d in Mt.devices()
            Mt.device!(d)
            Mt.synchronize()
        end
    else
        error("Unknown device for sync: $dev")
    end
    return nothing
end

function _reclaim_all_gpu_devices!(dev::Symbol)
    if dev === :cuda
        C = Main.CUDA
        cur = C.device()
        for d in C.devices()
            C.device!(d)
            C.reclaim()
        end
        C.device!(cur)
    elseif dev === :amdgpu
        M = Main.AMDGPU
        cur = M.device()
        for d in M.devices()
            M.device!(d)
            M.reclaim()
        end
        M.device!(cur)
    end
    return nothing
end

"""
    vendor_timing_sync!()

Synchronize the loaded GPU backend (used around vendor timing fences).
"""
function vendor_timing_sync!()
    _vendor_synchronize_for_device(device_from_loaded())
end

function _vendor_synchronize_for_device(dev::Symbol)
    if dev === :cuda
        Main.CUDA.synchronize()
    elseif dev === :amdgpu
        Main.AMDGPU.synchronize()
    elseif dev === :oneapi
        Main.oneAPI.synchronize()
    elseif dev === :metal
        Main.Metal.synchronize()
    else
        error("Unknown device: $dev")
    end
end

"""
    vendor_gpu_reclaim!()

Reclaim memory on backends that support it (CUDA, AMDGPU); no-op on others.
"""
function vendor_gpu_reclaim!()
    _reclaim_all_gpu_devices!(device_from_loaded())
end

"""
    unsafe_free_dense_gpu!(A)

`unsafe_free!` for a dense GPU matrix (vendor baseline temporaries).
"""
function unsafe_free_dense_gpu!(A::AbstractArray)
    if A isa AbstractGPUArray
        unsafe_free!(A)
    end
    return nothing
end

"""
    spd_ones_dense_vendor(T, N; diag_scale=T(N), device_id=0)

Dense **`ones(N,N)`** on the active GPU vendor (**CUDA, AMDGPU, oneAPI, or Metal** — whatever
[`device_from_loaded`](@ref) reports), then the same diagonal boost as [`spd_ones_darray`](@ref).
Used to compare Dagger **multi-GPU** `DArray` Cholesky with **single-GPU**
`LinearAlgebra.cholesky!` / vendor LAPACK (**potrf**) on an identical SPD matrix.
"""
function spd_ones_dense_vendor(
    ::Type{T},
    N::Int;
    diag_scale=nothing,
    device_id::Int=0,
) where {T<:AbstractFloat}
    N > 0 || throw(ArgumentError("N must be positive"))
    dscale = something(diag_scale, T(N))::T
    dev = device_from_loaded()
    δ = dscale - one(T)
    A = _vendor_dense_ones(T, N, dev, device_id)
    _diag_boost!(A, δ)
    return A
end

"""
    bench_vendor_cholesky_once!(A::AbstractMatrix)

**In-place** `cholesky!` on lower-triangular storage (`Hermitian(A, :L)`), i.e. the vendor GPU
LAPACK **potrf** path provided by the loaded GPU array package. **Overwrites** `A`; callers should
restore the matrix between timed runs (e.g. `copyto!` from a template) for repeated factorizations.
"""
function bench_vendor_cholesky_once!(A::AbstractMatrix{T}) where {T<:AbstractFloat}
    LinearAlgebra.cholesky!(Hermitian(A, :L))
    _vendor_synchronize(A)
    return nothing
end

"""
    spd_ones_darray([T=Float32], N, block_size, assignment_matrix; diag_scale = T(N))

Allocate `N×N` `ones` with `assignment`, then add `(diag_scale - 1)` on the global diagonal
(in-place on diagonal tiles only).
"""
function spd_ones_darray(
    ::Type{T},
    N::Int,
    block_size::Int,
    assignment::Matrix{P};
    diag_scale::T=T(N),
) where {T<:AbstractFloat,P<:Dagger.Processor}
    N > 0 || throw(ArgumentError("N must be positive"))
    block_size > 0 || throw(ArgumentError("block_size must be positive"))
    rem(N, block_size) == 0 || throw(ArgumentError("block_size must divide N"))
    δ = diag_scale - one(T)
    DA = ones(Blocks(block_size, block_size), T, N, N; assignment=assignment)
    Ac = DA.chunks
    nb = size(Ac, 1)
    @assert size(Ac, 1) == size(Ac, 2) == N ÷ block_size
    Dagger.spawn_datadeps() do
        for k in 1:nb
            Dagger.@spawn _diag_boost!(Dagger.InOut(Ac[k, k]), δ)
        end
    end
    wait_darray!(DA)
    return DA
end

function wait_darray!(A::Dagger.DArray)
    for c in A.chunks
        fetch(c)
    end
    return nothing
end

"""
    sync_darray!(A::Dagger.DArray)

Wait for all Dagger tasks backing the DArray to complete, then synchronize
all devices for the **loaded** GPU backend (CUDA, AMDGPU, oneAPI, or Metal).
Unlike `wait_darray!`, this does not transfer chunk data to the host.
"""
function sync_darray!(A::Dagger.DArray)
    for c in A.chunks
        wait(c)
    end
    _sync_all_gpu_devices!(device_from_loaded())
    return nothing
end

function unsafe_free_darray!(A::Dagger.DArray)
    for c in A.chunks
        raw = fetch(c; raw=true)
        if raw isa AbstractGPUArray
            unsafe_free!(raw)
        end
    end
    return nothing
end

function _purge_gpu_memory!()
    # Drain Dagger's scheduler chunk cache (holds GPU arrays across runs)
    n = Dagger.clear_chunk_cache!()
    n > 0 && @info "purge: cleared $n CHUNK_CACHE entries"

    GC.gc(true)
    GC.gc(true)
    dev = device_from_loaded()
    _reclaim_all_gpu_devices!(dev)
    return nothing
end
const purge_gpu_memory! = _purge_gpu_memory!

"""
    bench_cholesky_once!(DA) -> nothing
    bench_cholesky_once!(DA, algo::Symbol) -> nothing

Run `cholesky(DA)` (real SPD symmetric DArray) and wait until factor chunks are ready.
Out-of-place Dagger path should not destroy `DA` (see Dagger tests).

When `algo` is supplied (`:rl`, `:rl_la`, `:ll`), the corresponding Dagger Cholesky variant
is selected via `Dagger.CHOLESKY_ALGORITHM`. Default (no `algo`) uses `:rl_la`.
"""
function bench_cholesky_once!(DA::Dagger.DArray{T,2}) where {T<:Real}
    bench_cholesky_once!(DA, :rl_la)
end

function bench_cholesky_once!(DA::Dagger.DArray{T,2}, algo::Symbol) where {T<:Real}
    algo in CHOLESKY_ALGORITHMS ||
        throw(ArgumentError("Unknown Cholesky algorithm: $algo. Use one of $CHOLESKY_ALGORITHMS"))
    F = Dagger.with(Dagger.CHOLESKY_ALGORITHM => algo) do
        Dagger.darray_cholesky!(copy(DA))
    end
    _sync_chol_factor!(F)
    return nothing
end

"""
    bench_cholesky_inplace_once!(DA_work, algo)

**Compute-only timed path**: run `darray_cholesky!` on `DA_work` (already
restored from template by the caller) and synchronize all GPUs.
No memory purge, no `copyto!` — the caller handles setup before the timer
and cleanup after it.
"""
function bench_cholesky_inplace_once!(
    DA_work::Dagger.DArray{T,2},
    algo::Symbol=:rl_la,
) where {T<:Real}
    algo in CHOLESKY_ALGORITHMS ||
        throw(ArgumentError("Unknown Cholesky algorithm: $algo. Use one of $CHOLESKY_ALGORITHMS"))
    F = Dagger.with(Dagger.CHOLESKY_ALGORITHM => algo) do
        Dagger.darray_cholesky!(DA_work)
    end
    _sync_chol_factor!(F)
    return nothing
end

function _sync_chol_factor!(F::Cholesky)
    _sync_if_darray(F.factors)
    return nothing
end

_sync_if_darray(A::Dagger.DArray) = sync_darray!(A)
_sync_if_darray(A::LinearAlgebra.AbstractTriangular) = _sync_if_darray(parent(A))
_sync_if_darray(A::LinearAlgebra.Adjoint) = _sync_if_darray(parent(A))
_sync_if_darray(A::LinearAlgebra.Transpose) = _sync_if_darray(parent(A))
_sync_if_darray(_) = nothing

function _dense_like_da_for_verify(::Type{T}, A_host::Matrix{T}) where {T<:Real}
    dev = device_from_loaded()
    if dev === :cuda
        return Main.CUDA.CuArray(A_host)
    elseif dev === :amdgpu
        return Main.AMDGPU.ROCArray(A_host)
    elseif dev === :oneapi
        return Main.oneAPI.oneArray(A_host)
    elseif dev === :metal
        return Main.Metal.MtlArray(A_host)
    end
    error("Cannot build dense GPU matrix for verify on device: $dev")
end

"""
    verify_cholesky_small(DA; rtol=1e-4) -> Bool

Vendor (single-GPU) reference for correctness checking.
"""
function verify_cholesky_small(DA::Dagger.DArray{T,2}; rtol::Real=1e-4) where {T<:Real}
    Fd = Dagger.darray_cholesky!(copy(DA))
    _sync_chol_factor!(Fd)
    Ld = collect(Fd.L)
    A_host = collect(DA)
    A_gpu = _dense_like_da_for_verify(T, A_host)
    Fc = LinearAlgebra.cholesky!(LinearAlgebra.Hermitian(A_gpu, :L))
    _vendor_synchronize(A_gpu)
    Lc = Array(Fc.L)
    return isapprox(Ld, Lc; rtol=rtol)
end

end # module
