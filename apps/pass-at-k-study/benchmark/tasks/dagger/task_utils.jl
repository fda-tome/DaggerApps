# Loaded before every sample evaluation
using Dagger, LinearAlgebra, Statistics

# Optional GPU backends — must be at top level (using not allowed inside functions)
try
    using oneAPI
catch
end
try
    using CUDA
catch
end

# Detect available GPU backends gracefully
function available_gpu_scopes(n::Int=4)
    # Try oneAPI (Intel), then CUDA (NVIDIA), then fall back to CPU threads
    if isdefined(Main, :oneAPI)
        return [Dagger.scope(intel_gpu=i) for i in 1:min(n, length(Main.oneAPI.devices()))]
    end
    if isdefined(Main, :CUDA)
        return [Dagger.scope(cuda_gpu=i) for i in 1:min(n, length(Main.CUDA.devices()))]
    end
    # CPU fallback for smoke testing without GPUs
    return [Dagger.scope(thread=i) for i in 1:min(n, Threads.nthreads())]
end

GPU_SCOPES = available_gpu_scopes(4)

# Reference GEMM for correctness checking (CPU, exact)
function reference_gemm(A::Matrix{T}, B::Matrix{T}) where T
    A * B
end

# Measure median TFLOPS over n_trials
function measure_tflops(f, N::Int, n_trials::Int=5)
    flops = 2.0 * N^3
    times = [(@elapsed f(N)) for _ in 1:n_trials]
    flops / median(times) / 1e12
end
