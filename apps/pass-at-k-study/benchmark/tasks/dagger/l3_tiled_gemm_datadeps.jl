module L3TiledGemmDatadeps
const LEVEL = 3
const NAME  = "l3_tiled_gemm_datadeps"
const DESCRIPTION = "In-place tiled GEMM using spawn_datadeps with Out/In annotations"

const PROMPT = """
Using Dagger.jl, write a Julia function `datadeps_gemm(N::Int, block_size::Int)` that:

1. Creates Float32 DArrays DA (random), DB (random), DC (zeros), all N×N with Blocks(block_size, block_size)
2. Uses `Dagger.spawn_datadeps()` to perform in-place matrix multiplication:
       mul!(Out(DC), In(DA), In(DB))
3. Waits for completion (hint: collect DC)
4. Returns `collect(DC)` as Matrix{Float32}

Important rules:
- You MUST use `Dagger.spawn_datadeps()` with `Out()` and `In()` annotations
- Do NOT use fetch/wait inside the spawn_datadeps block
- Do NOT use `@sync` inside the spawn_datadeps block
- Use `LinearAlgebra.mul!` for the multiply

Function signature:
    datadeps_gemm(N::Int, block_size::Int) -> Matrix{Float32}
"""

const REFERENCE_SOLUTION = """
function datadeps_gemm(N::Int, block_size::Int)
    DA = rand(Blocks(block_size, block_size), Float32, N, N)
    DB = rand(Blocks(block_size, block_size), Float32, N, N)
    DC = zeros(Blocks(block_size, block_size), Float32, N, N)
    Dagger.spawn_datadeps() do
        Dagger.@spawn mul!(Out(DC), In(DA), In(DB))
    end
    collect(DC)
end
"""

const TEST_CASES = [
    (N=512,  block_size=128, tol=1e-2, check=:correctness),
    (N=1024, block_size=256, tol=1e-2, check=:correctness),
    (N=2048, block_size=512, tol=1e-2, check=:shape_only),
]

const ENTRY_FUNCTION = :datadeps_gemm
const PERF_TARGET_TFLOPS = 2.0
end
