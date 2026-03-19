module L1BasicSpawn
const LEVEL = 1
const NAME  = "l1_basic_spawn"
const DESCRIPTION = "Spawn 4 independent tasks on different GPU scopes and collect results"

const PROMPT = """
Using Dagger.jl, write a Julia function `parallel_sum(arrays::Vector{Matrix{Float32}})` that:
1. Takes a vector of exactly 4 Float32 matrices
2. Spawns one Dagger task per matrix using `Dagger.@spawn` to compute `sum(m)` on it
3. Pins each task to a different GPU scope from a pre-defined `GPU_SCOPES` vector (already in scope, length 4)
4. Collects all 4 results and returns them as a `Vector{Float32}`

Constraints:
- Use `Dagger.@spawn` not `@async` or `Threads.@spawn`
- Use `fetch()` to retrieve results
- Do not use `Dagger.with_options` — use the `scope=` keyword on `@spawn` directly

Return type must be `Vector{Float32}`.
"""

const REFERENCE_SOLUTION = """
function parallel_sum(arrays::Vector{Matrix{Float32}})
    tasks = [Dagger.@spawn scope=GPU_SCOPES[i] sum(arrays[i]) for i in 1:4]
    return Float32[fetch(t) for t in tasks]
end
"""

const TEST_CASES = [
    (sizes=[(128,128),(256,256),(64,64),(512,512)], check=:values),
    (sizes=[(1024,1024),(1024,1024),(1024,1024),(1024,1024)], check=:values),
]

const ENTRY_FUNCTION = :parallel_sum
const PERF_TARGET_TFLOPS = 0.0
end
