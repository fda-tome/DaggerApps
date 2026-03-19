module L4CustomScope
const LEVEL = 4
const NAME  = "l4_custom_scope"
const DESCRIPTION = "Define a custom GPU scope and run a small GEMM on it"

const PROMPT = """
Using Dagger.jl, write a Julia function `custom_scope_gemm(N::Int)` that:

1. Builds a custom scope that combines the first two entries of `GPU_SCOPES` using `Dagger.UnionScope`
2. Uses `Dagger.with_options(; scope=...)` to run inside that combined scope
3. Creates two random Float32 matrices A and B of size N×N (as DArrays or via @spawn), multiplies them, and returns `collect` of the result as Matrix{Float32}

Function signature:
    custom_scope_gemm(N::Int) -> Matrix{Float32}
"""

const REFERENCE_SOLUTION = """
function custom_scope_gemm(N::Int)
    custom = reduce(Dagger.UnionScope, GPU_SCOPES[1:2])
    Dagger.with_options(; scope=custom) do
        DA = rand(Blocks(N÷2, N÷2), Float32, N, N)
        DB = rand(Blocks(N÷2, N÷2), Float32, N, N)
        collect(DA * DB)
    end
end
"""

const TEST_CASES = [
    (N=256,  tol=1e-2, check=:correctness),
    (N=512,  tol=1e-2, check=:correctness),
]

const ENTRY_FUNCTION = :custom_scope_gemm
const PERF_TARGET_TFLOPS = 0.0
end
