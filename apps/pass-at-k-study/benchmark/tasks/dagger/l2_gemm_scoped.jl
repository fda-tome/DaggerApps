module L2GemmScoped
const LEVEL = 2
const NAME  = "l2_gemm_scoped"
const DESCRIPTION = "Distributed GEMM across 4 GPU scopes using DArray"

const PROMPT = """
Using Dagger.jl, write a Julia function `distributed_gemm(N::Int)` that:
1. Creates two random Float32 DArrays `DA` and `DB`, each of size N×N
2. Partitions both with `Blocks(N÷4, N÷4)` (4 blocks per dimension = 16 total)
3. Distributes blocks across the 4 GPU scopes in `GPU_SCOPES` (already in scope)
   using `Dagger.with_options(; scope=Dagger.scope(cuda_gpus=:))` or equivalent
4. Performs matrix multiplication `DC = DA * DB`
5. Returns `collect(DC)` as a plain `Matrix{Float32}`

The function signature must be:
    distributed_gemm(N::Int) -> Matrix{Float32}
"""

const REFERENCE_SOLUTION = """
function distributed_gemm(N::Int)
    all_scope = reduce(Dagger.UnionScope, GPU_SCOPES)
    Dagger.with_options(; scope=all_scope) do
        DA = rand(Blocks(N÷4, N÷4), Float32, N, N)
        DB = rand(Blocks(N÷4, N÷4), Float32, N, N)
        DC = DA * DB
        collect(DC)
    end
end
"""

const TEST_CASES = [
    (N=512,  tol=1e-2, check=:correctness),
    (N=1024, tol=1e-2, check=:correctness),
    (N=2048, tol=1e-2, check=:shape_only),   # too slow to verify exactly
]

const ENTRY_FUNCTION = :distributed_gemm
const PERF_TARGET_TFLOPS = 1.0   # should beat 1 TFLOP on 4 GPUs
end
