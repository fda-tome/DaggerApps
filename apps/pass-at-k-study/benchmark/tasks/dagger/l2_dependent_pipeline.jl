module L2DependentPipeline
const LEVEL = 2
const NAME  = "l2_dependent_pipeline"
const DESCRIPTION = "Chain of dependent Dagger tasks: rand -> normalize -> sum"

const PROMPT = """
Using Dagger.jl, write a Julia function `gpu_pipeline(N::Int)` that implements
a 3-stage pipeline of dependent tasks using `Dagger.@spawn`:

Stage 1: Generate a random Float32 matrix of size N×N (scoped to GPU_SCOPES[1])
Stage 2: Normalize it by dividing by its maximum value (depends on stage 1, scoped to GPU_SCOPES[2])
Stage 3: Compute the sum of the normalized matrix (depends on stage 2, scoped to GPU_SCOPES[3])

Rules:
- Each stage must be a separate `Dagger.@spawn` call
- Stages must be chained by passing the task handle of stage N as input to stage N+1
- Use `fetch()` only once at the very end to get the final scalar result
- Return the final Float32 scalar

Function signature:
    gpu_pipeline(N::Int) -> Float32
"""

const REFERENCE_SOLUTION = """
function gpu_pipeline(N::Int)
    t1 = Dagger.@spawn scope=GPU_SCOPES[1] rand(Float32, N, N)
    t2 = Dagger.@spawn scope=GPU_SCOPES[2] (A -> A ./ maximum(A))(t1)
    t3 = Dagger.@spawn scope=GPU_SCOPES[3] sum(t2)
    return fetch(t3)
end
"""

const TEST_CASES = [
    (N=256,  check=:scalar_finite),
    (N=1024, check=:scalar_finite),
    (N=4096, check=:scalar_finite),
]

const ENTRY_FUNCTION = :gpu_pipeline
const PERF_TARGET_TFLOPS = 0.0
end
