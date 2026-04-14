module L4DistributedSolverStep
const LEVEL = 4
const NAME = "l4_distributed_solver_step"
const DESCRIPTION = "Distributed iterative solver step"

const PROMPT = """
Legacy prompt field retained for compatibility. Prompt text is sourced from benchmark/prompts/shared_tasks.json.
"""

const REFERENCE_SOLUTION = """
function distributed_solver_step(N::Int)
    t1 = Dagger.@spawn rand(Float32, N, N)
    t2 = Dagger.@spawn (A -> (A .+ circshift(A, (1, 0)) .+ circshift(A, (-1, 0)) .+
                              circshift(A, (0, 1)) .+ circshift(A, (0, -1))) ./ 5f0)(t1)
    return fetch(t2)
end
"""

const TEST_CASES = [
    (N=64, check=:shape_only),
    (N=256, check=:shape_only),
]

const ENTRY_FUNCTION = :distributed_solver_step
const PERF_TARGET_TFLOPS = 0.0
end
