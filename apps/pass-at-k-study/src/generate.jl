#!/usr/bin/env julia
# Phase 1: LLM generation. Discover tasks per framework, call LLM API, write raw outputs to outputs/generated/*.jsonl.
# Never regenerate — outputs are frozen.

using HTTP
using JSON3
using ProgressMeter

const REPO_ROOT = dirname(@__DIR__)
const TASKS_DIR = joinpath(REPO_ROOT, "benchmark", "tasks")
const PROMPTS_DIR = joinpath(REPO_ROOT, "benchmark", "prompts")
const OUTPUT_DIR = joinpath(REPO_ROOT, "outputs", "generated")

function discover_tasks()::Vector{NamedTuple}
    tasks = NamedTuple[]
    for framework in readdir(TASKS_DIR)
        fw_path = joinpath(TASKS_DIR, framework)
        isdir(fw_path) || continue
        # Non-Julia frameworks: discover from tasks.json
        manifest_path = joinpath(fw_path, "tasks.json")
        if isfile(manifest_path)
            data = JSON3.read(read(manifest_path, String))
            list = data isa AbstractVector ? data : [data]
            for item in list
                name = String(get(item, :name, get(item, :id, "")))
                task_file = String(get(item, :task_file, name * ".cpp"))
                prompt = String(get(item, :prompt, ""))
                if haskey(item, :prompt_path)
                    prompt_path = joinpath(fw_path, String(item.prompt_path))
                    prompt = isfile(prompt_path) ? strip(read(prompt_path, String)) : prompt
                end
                isempty(name) && continue
                push!(tasks, (; framework = framework, task_file = task_file, name = name, prompt = prompt))
            end
            continue
        end
        # Dagger (and any .jl-based): discover from const NAME / PROMPT
        for f in sort(readdir(fw_path))
            (endswith(f, ".jl") && f != "task_utils.jl") || continue
            path = joinpath(fw_path, f)
            text = read(path, String)
            name_m = match(r"const NAME\s*=\s*\"([^\"]+)\"", text)
            prompt_m = match(r"const PROMPT = \"\"\"(.*?)\"\"\""s, text)
            if name_m !== nothing && prompt_m !== nothing
                push!(tasks, (;
                    framework = framework,
                    task_file = f,
                    name = strip(name_m.captures[1]),
                    prompt = strip(prompt_m.captures[1]),
                ))
            end
        end
    end
    tasks
end

function load_system_prompt()::String
    path = joinpath(PROMPTS_DIR, "system_prompt.txt")
    strip(read(path, String))
end

function load_cot_suffix(use_cot::Bool)::String
    use_cot || return ""
    path = joinpath(PROMPTS_DIR, "chain_of_thought_suffix.txt")
    isfile(path) || return ""
    "\n\n" * strip(read(path, String))
end

function call_llm(
    api_base::String,
    api_key::String,
    model::String,
    system_prompt::String,
    user_message::String;
    temperature::Float64 = 0.2,
    max_tokens::Int = 4096,
)::String
    url = endswith(api_base, '/') ? api_base * "chat/completions" : api_base * "/chat/completions"
    body = Dict(
        "model" => model,
        "messages" => [
            Dict("role" => "system", "content" => system_prompt),
            Dict("role" => "user", "content" => user_message),
        ],
        "temperature" => temperature,
        "max_tokens" => max_tokens,
    )
    headers = ["Content-Type" => "application/json", "Authorization" => "Bearer " * api_key]
    resp = HTTP.post(url, headers, JSON3.write(body))
    resp.status != 200 && error("API error: ", resp.status, " ", String(resp.body))
    data = JSON3.read(String(resp.body))
    content = data.choices[1].message.content
    content === nothing ? "" : String(content)
end

function run_generation(;
    model::String = "gpt-4o-mini",
    n_samples::Int = 5,
    task_filter::Union{String,Nothing} = nothing,
    api_base::Union{String,Nothing} = nothing,
    api_key::Union{String,Nothing} = nothing,
    use_cot::Bool = false,
    output_file::Union{String,Nothing} = nothing,
)
    mkpath(OUTPUT_DIR)
    base = something(api_base, "https://api.openai.com/v1")
    key = something(api_key, get(ENV, "OPENAI_API_KEY", "ollama"))
    out_path = something(output_file, joinpath(OUTPUT_DIR, "generated_" * replace(model, '/' => '_') * ".jsonl"))

    system_prompt = load_system_prompt()
    cot_suffix = load_cot_suffix(use_cot)
    task_list = discover_tasks()
    if task_filter !== nothing
        task_list = filter(t -> t.name == task_filter, task_list)
        isempty(task_list) && error("No task named ", repr(task_filter), ". Choices: ", [t.name for t in discover_tasks()])
    end

    open(out_path, "w") do io
        @showprogress "Tasks" for task in task_list
            for sample_id in 0:(n_samples - 1)
                user_message = task.prompt * cot_suffix
                response = try
                    call_llm(base, key, model, system_prompt, user_message)
                catch e
                    "[ERROR: $(sprint(showerror, e))]"
                end
                record = Dict(
                    "task" => task.name,
                    "task_file" => task.task_file,
                    "framework" => task.framework,
                    "model" => model,
                    "sample_id" => sample_id,
                    "response" => response,
                )
                println(io, JSON3.write(record))
            end
        end
    end
    println("Wrote ", out_path)
    out_path
end

function parse_cli(args::Vector{String})
    kwargs = Dict{Symbol,Any}(
        :model => "gpt-4o-mini",
        :n_samples => 5,
        :api_base => nothing,
        :api_key => nothing,
        :use_cot => false,
        :output_file => nothing,
        :task_filter => nothing,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--model" && i + 1 <= length(args)
            kwargs[:model] = args[i + 1]; i += 2
        elseif a == "--n-samples" && i + 1 <= length(args)
            kwargs[:n_samples] = parse(Int, args[i + 1]); i += 2
        elseif a == "--api-base" && i + 1 <= length(args)
            kwargs[:api_base] = args[i + 1]; i += 2
        elseif a == "--api-key" && i + 1 <= length(args)
            kwargs[:api_key] = args[i + 1]; i += 2
        elseif a == "--use-cot"
            kwargs[:use_cot] = true; i += 1
        elseif a == "--output" && i + 1 <= length(args)
            kwargs[:output_file] = args[i + 1]; i += 2
        elseif a == "--task" && i + 1 <= length(args)
            kwargs[:task_filter] = args[i + 1]; i += 2
        else
            i += 1
        end
    end
    kwargs
end

function main()
    kwargs = parse_cli(ARGS)
    run_generation(;
        model = kwargs[:model],
        n_samples = kwargs[:n_samples],
        task_filter = kwargs[:task_filter],
        api_base = kwargs[:api_base],
        api_key = kwargs[:api_key],
        use_cot = kwargs[:use_cot],
        output_file = kwargs[:output_file],
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
