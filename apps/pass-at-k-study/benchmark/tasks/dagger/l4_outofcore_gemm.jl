module L4OutofcoreGemm
const LEVEL = 4
const NAME  = "l4_outofcore_gemm"
const DESCRIPTION = "GEMM that explicitly streams tiles to avoid OOM — owner-computes pattern"

const PROMPT = """
Using Dagger.jl, implement a Julia function `streaming_gemm(N::Int, block_size::Int)`
that performs C = A * B using an explicit owner-computes tile loop:

For each output tile C[i,j], accumulate contributions from all k tiles:
    C[i,j] += A[i,k] * B[k,j]

Requirements:
1. Allocate DA, DB as random Float32 DArrays (N×N, Blocks(block_size, block_size))
   with round-robin GPU_SCOPES assignment (same formula as: GPU_SCOPES[(i+j-2)%4+1])
2. Allocate DC as zeros with the same assignment
3. Use `Dagger.spawn_datadeps()` wrapping an explicit triple loop over (i, j, k)
   where each inner step calls:
       Dagger.@spawn LinearAlgebra.mul!(Out(DC[i,j]), In(DA[i,k]), In(DB[k,j]), 1f0, 1f0)
   (the 1f0, 1f0 are alpha/beta for accumulation)
4. Return collect(DC)

Note: DC[i,j] notation on a DArray selects the (i,j)-th block chunk.
Use `DArray` block indexing: `DA.chunks[i,k]` if direct block access is needed.

Function signature:
    streaming_gemm(N::Int, block_size::Int) -> Matrix{Float32}
"""

const REFERENCE_SOLUTION = """
function streaming_gemm(N::Int, block_size::Int)
    n_blk  = N ÷ block_size
    asgn   = [GPU_SCOPES[(i+j-2)%4+1] for i in 1:n_blk, j in 1:n_blk]
    DA = rand(Blocks(block_size, block_size), Float32, N, N; assignment=asgn)
    DB = rand(Blocks(block_size, block_size), Float32, N, N; assignment=asgn)
    DC = zeros(Blocks(block_size, block_size), Float32, N, N; assignment=asgn)
    Dagger.spawn_datadeps() do
        for i in 1:n_blk, j in 1:n_blk
            for k in 1:n_blk
                Dagger.@spawn mul!(Out(DC.chunks[i,j]), In(DA.chunks[i,k]),
                                   In(DB.chunks[k,j]), 1f0, 1f0)
            end
        end
    end
    collect(DC)
end
"""

const TEST_CASES = [
    (N=512,  block_size=128, tol=1e-2, check=:correctness),
    (N=1024, block_size=256, tol=1e-2, check=:correctness),
]

const ENTRY_FUNCTION = :streaming_gemm
const PERF_TARGET_TFLOPS = 2.0
end
