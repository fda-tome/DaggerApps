cd(ENV["PROJ"])
using CUDA, Dagger, LinearAlgebra
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
Dagger.clear_chunk_cache!()
GC.gc(true); GC.gc(true)
for d in CUDA.devices(); CUDA.device!(d); CUDA.reclaim(); end

# 2) CUDA kernel precompile at actual tile size
CUDA.device!(0)
T = Float32
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

# 3) Run benchmark (0 warmup, 1 trial — kernels already compiled)
run_benchmark()
