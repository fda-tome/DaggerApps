module L1DArrayCreate
const LEVEL = 1
const NAME  = "l1_darray_create"
const DESCRIPTION = "Create a distributed DArray with Blocks and collect it back to CPU"

const PROMPT = """
Using Dagger.jl, write a Julia function `make_darray(N::Int, block_size::Int)` that:
1. Creates a random Float32 DArray of size N×N partitioned into blocks of size block_size×block_size
2. Uses `Dagger.Blocks(block_size, block_size)` for the partition spec
3. Collects the DArray back to a plain CPU `Matrix{Float32}` and returns it

The function signature must be:
    make_darray(N::Int, block_size::Int) -> Matrix{Float32}

Do not pin to any specific GPU scope. Use default scheduling.
"""

const REFERENCE_SOLUTION = """
function make_darray(N::Int, block_size::Int)
    DA = rand(Blocks(block_size, block_size), Float32, N, N)
    return collect(DA)
end
"""

const TEST_CASES = [
    (N=512,  block_size=128, check=:shape_and_type),
    (N=1024, block_size=256, check=:shape_and_type),
    (N=2048, block_size=512, check=:shape_and_type),
]

const ENTRY_FUNCTION = :make_darray
const PERF_TARGET_TFLOPS = 0.0
end
