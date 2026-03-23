module L3DependencyChainReduce
const LEVEL = 3
const NAME = "l3_dependency_chain_reduce"
const DESCRIPTION = "Multi-stage dependency chain with reduction"

const PROMPT = """
Legacy prompt field retained for compatibility. Prompt text is sourced from benchmark/prompts/shared_tasks.json.
"""

const REFERENCE_SOLUTION = """
function dependency_chain_reduce(N::Int)
    t1 = Dagger.@spawn rand(Float32, N, N)
    t2 = Dagger.@spawn (A -> A .* 2f0)(t1)
    t3 = Dagger.@spawn sum(t2)
    return Float32(fetch(t3))
end
"""

const TEST_CASES = [
    (N=128, check=:scalar_finite),
    (N=512, check=:scalar_finite),
]

const ENTRY_FUNCTION = :dependency_chain_reduce
const PERF_TARGET_TFLOPS = 0.0
end
