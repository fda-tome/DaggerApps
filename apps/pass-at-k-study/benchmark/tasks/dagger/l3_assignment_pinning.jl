module L3AssignmentPinning
const LEVEL = 3
const NAME  = "l3_assignment_pinning"
const DESCRIPTION = "Pin DArray blocks to specific GPUs using the assignment= argument"

const PROMPT = """
Using Dagger.jl, write a Julia function `pinned_gemm(N::Int, block_size::Int)` that:

1. Computes n_blocks = N ÷ block_size
2. Creates a 2D assignment matrix where block [i,j] is pinned to
   GPU_SCOPES[(i + j - 2) % 4 + 1]  (round-robin across 4 GPUs)
3. Allocates Float32 DArrays DA (random), DB (random), DC (zeros),
   each N×N with Blocks(block_size, block_size) and the assignment above
4. Performs `DC = DA * DB` (out-of-place multiply)
5. Returns collect(DC) as Matrix{Float32}

The `assignment` keyword expects a matrix of `Dagger.ThreadProc` (or other `Processor`) values with shape (n_blocks, n_blocks). Use worker id `1` and thread ids `1:4` in the same round-robin pattern as `GPU_SCOPES[(i + j - 2) % 4 + 1]` would imply for CPU thread fallback.

Function signature:
    pinned_gemm(N::Int, block_size::Int) -> Matrix{Float32}
"""

const REFERENCE_SOLUTION = """
function pinned_gemm(N::Int, block_size::Int)
    n_blocks = N ÷ block_size
    assignment = [Dagger.ThreadProc(1, mod1(i + j - 1, 4))
                  for i in 1:n_blocks, j in 1:n_blocks]
    DA = rand(Blocks(block_size, block_size), Float32, N, N; assignment=assignment)
    DB = rand(Blocks(block_size, block_size), Float32, N, N; assignment=assignment)
    DC = DA * DB
    collect(DC)
end
"""

const TEST_CASES = [
    (N=512,  block_size=128, tol=1e-2, check=:correctness),
    (N=1024, block_size=256, tol=1e-2, check=:correctness),
]

const ENTRY_FUNCTION = :pinned_gemm
const PERF_TARGET_TFLOPS = 2.0
end
