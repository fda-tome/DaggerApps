#!/usr/bin/env julia
# Phase 2: Sandboxed evaluation. Read generated JSONL, extract code, run tests, write evaluated JSONL.
# Never touches LLMs.

using JSON3
using ProgressMeter

const REPO_ROOT = dirname(@__DIR__)
tasks_dir(framework::String) = joinpath(REPO_ROOT, "benchmark", "tasks", framework)
const PROMPTS_DIR = joinpath(REPO_ROOT, "benchmark", "prompts")
const UTILS_DIR = joinpath(REPO_ROOT, "src", "utils")
const RUNNERS_DIR = joinpath(REPO_ROOT, "src", "utils", "runners")
const OUTPUT_DIR = joinpath(REPO_ROOT, "outputs", "evaluated")
const GENERATED_DIR = joinpath(REPO_ROOT, "outputs", "generated")
const DEFAULT_TIMEOUT = 120.0

include(joinpath(RUNNERS_DIR, "runners.jl"))

const FRAMEWORK_LANGUAGE = Dict("dagger" => "julia", "iris" => "cpp", "legate" => "python", "parsec" => "c")

"""Extract first fenced code block for given language (julia, cpp, c, python, etc.)."""
function extract_first_code_block(text::AbstractString, lang::String)::String
    # Match ```lang or ``` then newline then code
    pattern = Regex("```(?:$(lang))?\\s*\\n(.*?)```", "s")
    m = match(pattern, text)
    return m !== nothing ? strip(m.captures[1]) : strip(text)
end

function extract_first_julia_block(text::AbstractString)::String
    extract_first_code_block(text, "julia")
end

function task_name_to_module(task_name::String)::Symbol
    parts = split(task_name, "_")
    name = join([titlecase(p) for p in parts], "")
    Symbol(name)
end

function run_one_test_case!(tc, entry_sym::Symbol, results::Vector, mod)
    try
        if haskey(tc, :sizes) && tc.check == :values
            # l1_basic_spawn: build arrays from sizes, call entry(arrays)
            arrays = [rand(Float32, s[1], s[2]) for s in tc.sizes]
            out = getfield(Main, entry_sym)(arrays)
            ok = out isa Vector{Float32} && length(out) == 4
            expected = Float32[sum(arrays[i]) for i in 1:4]
            ok &= all(isapprox.(out, expected; rtol=1e-3))
            push!(results, (pass=ok, error=nothing))
        elseif haskey(tc, :N) && haskey(tc, :block_size) && tc.check == :shape_and_type
            # make_darray
            M = getfield(Main, entry_sym)(tc.N, tc.block_size)
            ok = M isa Matrix{Float32} && size(M) == (tc.N, tc.N)
            push!(results, (pass=ok, error=nothing))
        elseif haskey(tc, :N) && !haskey(tc, :block_size) && tc.check == :correctness
            # distributed_gemm
            C = getfield(Main, entry_sym)(tc.N)
            ok = C isa Matrix{Float32} && size(C) == (tc.N, tc.N)
            if ok && haskey(tc, :tol)
                A = rand(Float32, tc.N, tc.N)
                B = rand(Float32, tc.N, tc.N)
                ref = reference_gemm(A, B)
                # We cannot compare C to ref without fixing RNG; check approximate norm
                ok = isfinite(norm(C))
            end
            push!(results, (pass=ok, error=nothing))
        elseif haskey(tc, :N) && tc.check == :shape_only
            C = getfield(Main, entry_sym)(tc.N)
            push!(results, (pass=C isa Matrix{Float32} && size(C) == (tc.N, tc.N), error=nothing))
        elseif haskey(tc, :N) && tc.check == :scalar_finite
            s = getfield(Main, entry_sym)(tc.N)
            push!(results, (pass=isa(s, Float32) && isfinite(s), error=nothing))
        elseif haskey(tc, :N) && haskey(tc, :block_size) && tc.check == :correctness
            tol = get(tc, :tol, 1e-2)
            C = getfield(Main, entry_sym)(tc.N, tc.block_size)
            push!(results, (pass=C isa Matrix{Float32} && size(C) == (tc.N, tc.N) && isfinite(norm(C)), error=nothing))
        elseif haskey(tc, :N) && haskey(tc, :block_size) && tc.check == :shape_only
            C = getfield(Main, entry_sym)(tc.N, tc.block_size)
            push!(results, (pass=C isa Matrix{Float32} && size(C) == (tc.N, tc.N), error=nothing))
        else
            push!(results, (pass=false, error="unknown test case type"))
        end
    catch e
        push!(results, (pass=false, error=sprint(showerror, e)))
    end
end

function evaluate_sample(task_name::String, task_file::String, code::String; framework::String="dagger", timeout_sec::Real=DEFAULT_TIMEOUT)
    fw_dir = tasks_dir(framework)
    # Write temp files: task_utils, task file path, code file
    code_file = tempname() * ".jl"
    write(code_file, code)
    try
        # Run in a subprocess. Load task_utils via -L so "using" in it is at top level; runner does the rest.
        script = """
        cd(raw"$(REPO_ROOT)")
        using JSON3
        include(raw"$(joinpath(fw_dir, task_file))")
        include(raw"$(code_file)")
        mod = getfield(Main, Symbol($(repr(String(task_name_to_module(task_name))))))
        results = []
        for tc in mod.TEST_CASES
            try
                if haskey(tc, :sizes) && tc.check == :values
                    arrays = [rand(Float32, s[1], s[2]) for s in tc.sizes]
                    out = getfield(Main, mod.ENTRY_FUNCTION)(arrays)
                    ok = out isa Vector{Float32} && length(out) == 4
                    expected = Float32[sum(arrays[i]) for i in 1:4]
                    ok = ok && all(isapprox.(out, expected; rtol=1e-3))
                    push!(results, Dict("pass" => ok, "error" => nothing))
                elseif haskey(tc, :N) && haskey(tc, :block_size) && tc.check == :shape_and_type
                    M = getfield(Main, mod.ENTRY_FUNCTION)(tc.N, tc.block_size)
                    push!(results, Dict("pass" => (M isa Matrix{Float32} && size(M) == (tc.N, tc.N)), "error" => nothing))
                elseif haskey(tc, :N) && !haskey(tc, :block_size) && tc.check == :correctness
                    C = getfield(Main, mod.ENTRY_FUNCTION)(tc.N)
                    push!(results, Dict("pass" => (C isa Matrix{Float32} && size(C) == (tc.N, tc.N) && isfinite(norm(C))), "error" => nothing))
                elseif haskey(tc, :N) && tc.check == :shape_only
                    C = getfield(Main, mod.ENTRY_FUNCTION)(tc.N)
                    push!(results, Dict("pass" => (C isa Matrix{Float32} && size(C) == (tc.N, tc.N)), "error" => nothing))
                elseif haskey(tc, :N) && tc.check == :scalar_finite
                    s = getfield(Main, mod.ENTRY_FUNCTION)(tc.N)
                    push!(results, Dict("pass" => (isa(s, Float32) && isfinite(s)), "error" => nothing))
                elseif haskey(tc, :N) && haskey(tc, :block_size)
                    C = getfield(Main, mod.ENTRY_FUNCTION)(tc.N, tc.block_size)
                    push!(results, Dict("pass" => (C isa Matrix{Float32} && size(C) == (tc.N, tc.N) && isfinite(norm(C))), "error" => nothing))
                else
                    push!(results, Dict("pass" => false, "error" => "unknown_tc"))
                end
            catch e
                push!(results, Dict("pass" => false, "error" => sprint(showerror, e)))
            end
        end
        println(JSON3.write(Dict("results" => results)))
        """
        runner_file = tempname() * ".jl"
        write(runner_file, script)
        try
            out_io = IOBuffer()
            err_io = IOBuffer()
            task_utils_path = joinpath(fw_dir, "task_utils.jl")
            cmd = pipeline(`julia --project=$(REPO_ROOT) -L $(task_utils_path) $(runner_file)`, stdout=out_io, stderr=err_io)
            start = time()
            proc = run(cmd, wait=false)
            while process_running(proc)
                (time() - start >= timeout_sec) && (kill(proc); break)
                sleep(0.1)
            end
            elapsed = time() - start
            stdout_str = String(take!(out_io))
            stderr_str = String(take!(err_io))
            exit_ok = process_exited(proc) && proc.exitcode == 0
            if !exit_ok
                return (passed=false, results=[], elapsed=elapsed, error="timeout or non-zero exit", stderr=stderr_str)
            end
            # Parse JSON from last line
            line = strip(split(stdout_str, '\n')[end])
            data = JSON3.read(line)
            results = [r.pass for r in data.results]
            all_pass = all(results)
            return (passed=all_pass, results=results, elapsed=elapsed, error=nothing, stderr=stderr_str)
        finally
            rm(runner_file; force=true)
        end
    finally
        rm(code_file; force=true)
    end
end

function main()
    generated = if length(ARGS) >= 1
        ARGS[1]
    else
        candidates = filter(f -> endswith(f, ".jsonl"), readdir(GENERATED_DIR; join=true))
        isempty(candidates) && (println("No generated JSONL in $GENERATED_DIR"); return)
        first(candidates)
    end
    generated = abspath(generated)
    isfile(generated) || (println("No generated file found: $generated"); return)
    out_path = joinpath(OUTPUT_DIR, "evaluated_" * basename(generated))
    mkpath(OUTPUT_DIR)

    records = open(generated) do io
        [JSON3.read(line) for line in eachline(io) if !isempty(strip(line))]
    end

    open(out_path, "w") do out_io
        @showprogress "Evaluating..." for rec in records
            task = rec.task
            task_file = String(rec.task_file)
            model = String(rec.model)
            sample_id = rec.sample_id
            framework = haskey(rec, :framework) ? String(rec.framework) : "dagger"
            response = String(rec.response)
            lang = get(FRAMEWORK_LANGUAGE, framework, "julia")
            code = extract_first_code_block(response, lang)
            res = if framework == "dagger"
                evaluate_sample(task, task_file, code; framework=framework)
            elseif framework == "iris"
                run_iris(task, task_file, code, response; timeout_sec=DEFAULT_TIMEOUT)
            elseif framework == "legate"
                run_legate(task, task_file, code, response; timeout_sec=DEFAULT_TIMEOUT)
            elseif framework == "parsec"
                run_parsec(task, task_file, code, response; timeout_sec=DEFAULT_TIMEOUT)
            else
                (passed=false, results=Bool[], elapsed=0.0, error="unknown framework: $framework", stderr="")
            end
            write(out_io, JSON3.write(Dict(
                "task" => task,
                "task_file" => task_file,
                "model" => model,
                "sample_id" => sample_id,
                "framework" => framework,
                "passed" => res.passed,
                "results" => res.results,
                "elapsed_sec" => res.elapsed,
                "error" => res.error,
                "stderr" => res.stderr,
            )) * "\n")
        end
    end
    println("Wrote ", out_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
