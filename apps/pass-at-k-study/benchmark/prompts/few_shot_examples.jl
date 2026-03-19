# Few-shot examples (snippets to optionally prepend to prompts)
# Example: simple @spawn usage
"""
function example_parallel_sum(arrays::Vector{Matrix{Float32}})
    tasks = [Dagger.@spawn scope=GPU_SCOPES[i] sum(arrays[i]) for i in 1:4]
    return Float32[fetch(t) for t in tasks]
end
"""
