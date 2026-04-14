# Native (canonical) reference sources under benchmark/tasks/{dagger,iris,legate}/.
# Used to compare LLM-generated samples against the bundled implementations.

"""Extract the `REFERENCE_SOLUTION` string from a Dagger task module file."""
function extract_dagger_reference_solution(task_jl_path::String)::String
    text = read(task_jl_path, String)
    m = match(r"const\s+REFERENCE_SOLUTION\s*=\s*\"\"\"(.*?)\"\"\""s, text)
    m === nothing && error("missing const REFERENCE_SOLUTION triple-quoted block in $task_jl_path")
    return String(strip(m.captures[1]))
end

"""Full source text of the native reference for this framework task file."""
function native_reference_code(repo_root::String, framework::String, task_file::String)::String
    path = joinpath(repo_root, "benchmark", "tasks", framework, task_file)
    isfile(path) || error("native reference file not found: $path")
    if framework == "dagger"
        return extract_dagger_reference_solution(path)
    end
    return read(path, String)
end
