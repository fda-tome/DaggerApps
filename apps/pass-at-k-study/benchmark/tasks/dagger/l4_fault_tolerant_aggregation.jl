module L4FaultTolerantAggregation
const LEVEL = 4
const NAME = "l4_fault_tolerant_aggregation"
const DESCRIPTION = "Spawn subtasks; ignore non-finite contributions; aggregate valid values"

const PROMPT = """
Legacy prompt field retained for compatibility. Prompt text is sourced from benchmark/prompts/shared_tasks.json.
"""

const REFERENCE_SOLUTION = """
function fault_contrib_l4(ii::Int, N::Int)::Float32
    v = (ii == 3) ? Float32(NaN) : Float32(ii * N)
    isfinite(v) ? v : zero(Float32)
end
function fault_tolerant_aggregate(N::Int)::Float32
    s = zero(Float32)
    for i in 1:4
        let ii = i
            t = Dagger.@spawn fault_contrib_l4(ii, N)
            s += fetch(t)
        end
    end
    return s
end
"""

const TEST_CASES = [
    (N=10, expected=70.0, check=:scalar_expected),
    (N=100, expected=700.0, check=:scalar_expected),
]

const ENTRY_FUNCTION = :fault_tolerant_aggregate
const PERF_TARGET_TFLOPS = 0.0
end
