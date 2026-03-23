#!/usr/bin/env julia
# Verify bundled native references for dagger / iris / legate (same tasks as tasks.json).
# Usage: julia --project=apps/pass-at-k-study scripts/verify_native_references.jl

using JSON3

# App root (parent of `scripts/`, same as `src/` parent).
const ROOT = dirname(@__DIR__)
const TASKS_ROOT = joinpath(ROOT, "benchmark", "tasks")

include(joinpath(ROOT, "src", "evaluate.jl"))

function verify_all()::Bool
    ok = true
    for fw in ("dagger", "iris", "legate")
        path = joinpath(TASKS_ROOT, fw, "tasks.json")
        list = JSON3.read(read(path, String))
        list = list isa AbstractVector ? list : [list]
        for item in list
            task = String(get(item, :id, get(item, :name, "")))
            task_file = String(get(item, :task_file, ""))
            isempty(task) && continue
            r = evaluate_native_reference(task, task_file, fw)
            st = r.passed ? "PASS" : "FAIL"
            println(rpad(fw, 8), " ", rpad(task, 38), " ", st)
            if !r.passed
                ok = false
                r.error !== nothing && println("         error: ", r.error)
                !isempty(r.stderr) && println("         stderr: ", first(r.stderr, min(500, length(r.stderr))))
            end
        end
    end
    ok
end

exit(verify_all() ? 0 : 1)
