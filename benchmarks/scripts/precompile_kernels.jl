cd(ENV["PROJ"])
using Dagger
using LinearAlgebra

# Load one GPU backend before Dagger (same rule as gpu-cholesky benchmark).
# CHOLESKY_PRECOMPILE_BACKEND=cuda|amdgpu (default cuda, for ALCF CUDA jobs)
const _BE = lowercase(strip(get(ENV, "CHOLESKY_PRECOMPILE_BACKEND", "cuda")))
if _BE == "amdgpu"
    using AMDGPU
    using GPUArrays: unsafe_free!
else
    using CUDA
end

include(joinpath(ENV["PROJ"], "DaggerApps", "benchmarks", "scripts", "gpu-cholesky.jl"))

bs = parse(Int, ENV["CHOLESKY_BLOCK"])

# 1) Dagger machinery precompile (tiny 128×128, 2×2 grid, no copy)
procs = DG.four_gpu_processors()
asg = DG.cholesky_block_cyclic_assignment(procs, 2)
for a in (:rl_la, :ll)
    DA = DG.spd_ones_darray(Float32, 128, 64, asg)
    Dagger.with(Dagger.CHOLESKY_ALGORITHM => a) do
        Dagger.darray_cholesky!(DA)
    end
    DG.unsafe_free_darray!(DA)
end
isdefined(Dagger, :clear_chunk_cache!) && Dagger.clear_chunk_cache!()
GC.gc(true); GC.gc(true)

if _BE == "cuda"
    for d in CUDA.devices(); CUDA.device!(d); CUDA.reclaim(); end
else
    M = AMDGPU
    cur = M.device()
    for d in M.devices()
        M.device!(d)
        M.reclaim()
    end
    M.device!(cur)
end

# 2) Vendor BLAS precompile at actual tile size
T = Float32
if _BE == "cuda"
    CUDA.device!(0)
    A = CUDA.rand(T, bs, bs)
    A .= A * A' + bs * I
    LinearAlgebra.cholesky!(LinearAlgebra.Hermitian(A, :L))
    L = LinearAlgebra.LowerTriangular(A)
    B = CUDA.rand(T, bs, bs)
    rdiv!(B, L')
    C = CUDA.rand(T, bs, bs)
    mul!(C, B, B', -one(T), one(T))
    CUDA.unsafe_free!(A); CUDA.unsafe_free!(B); CUDA.unsafe_free!(C)
    GC.gc(true); GC.gc(true)
    for d in CUDA.devices(); CUDA.device!(d); CUDA.reclaim(); end
else
    AMDGPU.device!(AMDGPU.devices()[1])
    A = AMDGPU.rand(T, bs, bs)
    A .= A * A' + bs * I
    LinearAlgebra.cholesky!(LinearAlgebra.Hermitian(A, :L))
    L = LinearAlgebra.LowerTriangular(A)
    B = AMDGPU.rand(T, bs, bs)
    rdiv!(B, L')
    C = AMDGPU.rand(T, bs, bs)
    mul!(C, B, B', -one(T), one(T))
    unsafe_free!(A); unsafe_free!(B); unsafe_free!(C)
    GC.gc(true); GC.gc(true)
    cur = AMDGPU.device()
    for d in AMDGPU.devices()
        AMDGPU.device!(d)
        AMDGPU.reclaim()
    end
    AMDGPU.device!(cur)
end

# 3) Run benchmark (0 warmup, 1 trial — kernels already compiled)
run_benchmark()
