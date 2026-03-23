module L3PartitionedExchangeValidate
const LEVEL = 3
const NAME = "l3_partitioned_exchange_validate"
const DESCRIPTION = "Partitioned exchange and reconstruction validation"

const PROMPT = """
Legacy prompt field retained for compatibility. Prompt text is sourced from benchmark/prompts/shared_tasks.json.
"""

const REFERENCE_SOLUTION = """
function partitioned_exchange_validate(N::Int)
    t1 = Dagger.@spawn rand(Float32, N, N)
    t2 = Dagger.@spawn (A -> A .+ permutedims(A))(t1)
    return fetch(t2)
end
"""

const TEST_CASES = [
    (N=128, check=:shape_only),
    (N=512, check=:shape_only),
]

const ENTRY_FUNCTION = :partitioned_exchange_validate
const PERF_TARGET_TFLOPS = 0.0
end
