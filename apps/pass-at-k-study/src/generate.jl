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
const SHARED_TASKS_PATH = joinpath(PROMPTS_DIR, "shared_tasks.json")
const RUNTIME_CONTEXTS_PATH = joinpath(PROMPTS_DIR, "runtime_contexts.json")
const DEFAULT_DOCS_CONTEXT_DIR = joinpath(PROMPTS_DIR, "docs_context")
const DEFAULT_DOCS_CONTEXT_INDEX = joinpath(DEFAULT_DOCS_CONTEXT_DIR, "index.json")

function build_prompt(prompt_template::String, runtime_ctx)::String
    out = prompt_template
    out = replace(out, "{{RUNTIME_NAME}}" => String(runtime_ctx.runtime_name))
    out = replace(out, "{{LANGUAGE_NAME}}" => String(runtime_ctx.language_name))
    out = replace(out, "{{CODE_FENCE_LANG}}" => String(runtime_ctx.code_fence_lang))
    out = replace(out, "{{RUNTIME_REPO_URL}}" => String(runtime_ctx.repo_url))
    out = replace(out, "{{RUNTIME_DOCS_URL}}" => String(runtime_ctx.docs_url))
    out
end

function load_docs_context_index(docs_context_dir::String)::Dict{String,Any}
    idx = joinpath(docs_context_dir, "index.json")
    isfile(idx) || return Dict{String,Any}()
    raw = JSON3.read(read(idx, String))
    out = Dict{String,Any}()
    for (k, v) in pairs(raw)
        out[String(k)] = v
    end
    out
end

function load_docs_snippet(docs_context_dir::String, rel_path::String)::Union{Dict{String,Any},Nothing}
    path = joinpath(docs_context_dir, rel_path)
    isfile(path) || return nothing
    raw = JSON3.read(read(path, String))
    out = Dict{String,Any}()
    for (k, v) in pairs(raw)
        out[String(k)] = v
    end
    out
end

function select_docs_snippets(
    docs_context_index::Dict{String,Any},
    docs_context_dir::String,
    framework::String,
    task_name::String;
    top_k::Int = 3,
)::Vector{Dict{String,Any}}
    frameworks = get(docs_context_index, "frameworks", nothing)
    frameworks === nothing && return Dict{String,Any}[]
    fw = get(frameworks, framework, nothing)
    fw === nothing && return Dict{String,Any}[]

    rel_files = String[]
    for p in get(fw, "common", String[])
        push!(rel_files, String(p))
    end
    tasks = get(fw, "tasks", Dict{String,Any}())
    for p in get(tasks, task_name, String[])
        push!(rel_files, String(p))
    end

    snippets = Dict{String,Any}[]
    for rel in rel_files
        s = load_docs_snippet(docs_context_dir, rel)
        s === nothing && continue
        push!(snippets, s)
    end
    isempty(snippets) && return snippets
    top_k <= 0 && return snippets
    snippets[1:min(top_k, length(snippets))]
end

function render_docs_block(snippets::Vector{Dict{String,Any}})::String
    isempty(snippets) && return ""
    lines = String["Relevant local documentation excerpts (offline context):"]
    for (i, s) in enumerate(snippets)
        title = String(get(s, "title", "Untitled excerpt"))
        source = String(get(s, "source_url", ""))
        excerpt = strip(String(get(s, "excerpt", "")))
        isempty(excerpt) && continue
        push!(lines, "")
        push!(lines, "[$i] " * title)
        isempty(source) || push!(lines, "Source: " * source)
        push!(lines, excerpt)
    end
    join(lines, "\n")
end

function load_shared_tasks()::Dict{String,Any}
    isfile(SHARED_TASKS_PATH) || error("Missing shared tasks file: ", SHARED_TASKS_PATH)
    raw = JSON3.read(read(SHARED_TASKS_PATH, String))
    arr = raw isa AbstractVector ? raw : [raw]
    table = Dict{String,Any}()
    for item in arr
        id = String(get(item, :id, ""))
        isempty(id) && continue
        table[id] = item
    end
    table
end

function load_runtime_contexts()::Dict{String,Any}
    isfile(RUNTIME_CONTEXTS_PATH) || error("Missing runtime contexts file: ", RUNTIME_CONTEXTS_PATH)
    raw = JSON3.read(read(RUNTIME_CONTEXTS_PATH, String))
    table = Dict{String,Any}()
    for (k, v) in pairs(raw)
        table[String(k)] = v
    end
    table
end

function discover_tasks()::Vector{NamedTuple}
    shared_tasks = load_shared_tasks()
    runtime_contexts = load_runtime_contexts()
    tasks = NamedTuple[]
    for framework in readdir(TASKS_DIR)
        fw_path = joinpath(TASKS_DIR, framework)
        isdir(fw_path) || continue
        haskey(runtime_contexts, framework) || continue
        runtime_ctx = runtime_contexts[framework]
        # Unified path: discover from tasks.json for every framework.
        manifest_path = joinpath(fw_path, "tasks.json")
        if isfile(manifest_path)
            data = JSON3.read(read(manifest_path, String))
            list = data isa AbstractVector ? data : [data]
            for item in list
                name = String(get(item, :name, get(item, :id, "")))
                task_file = String(get(item, :task_file, name * ".cpp"))
                shared = get(shared_tasks, name, nothing)
                shared === nothing && error("Task ", repr(name), " from ", manifest_path, " not found in ", SHARED_TASKS_PATH)
                prompt_template = String(get(shared, :prompt_template, ""))
                prompt = build_prompt(prompt_template, runtime_ctx)
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
                    prompt = build_prompt(strip(prompt_m.captures[1]), runtime_ctx),
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
    docs_context_dir::String = DEFAULT_DOCS_CONTEXT_DIR,
    docs_top_k::Int = 3,
    enable_docs_context::Bool = true,
)
    mkpath(OUTPUT_DIR)
    base = something(api_base, "https://api.openai.com/v1")
    key = something(api_key, get(ENV, "OPENAI_API_KEY", "ollama"))
    out_path = something(output_file, joinpath(OUTPUT_DIR, "generated_" * replace(model, '/' => '_') * ".jsonl"))

    system_prompt = load_system_prompt()
    cot_suffix = load_cot_suffix(use_cot)
    task_list = discover_tasks()
    docs_context_index = enable_docs_context ? load_docs_context_index(docs_context_dir) : Dict{String,Any}()
    if task_filter !== nothing
        task_list = filter(t -> t.name == task_filter, task_list)
        isempty(task_list) && error("No task named ", repr(task_filter), ". Choices: ", [t.name for t in discover_tasks()])
    end

    open(out_path, "w") do io
        @showprogress "Tasks" for task in task_list
            for sample_id in 0:(n_samples - 1)
                docs_block = ""
                if enable_docs_context
                    snippets = select_docs_snippets(
                        docs_context_index,
                        docs_context_dir,
                        task.framework,
                        task.name;
                        top_k = docs_top_k,
                    )
                    docs_block = render_docs_block(snippets)
                end
                user_message = isempty(docs_block) ? task.prompt * cot_suffix : task.prompt * "\n\n" * docs_block * cot_suffix
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
                    "prompt_used" => user_message,
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
        :docs_context_dir => DEFAULT_DOCS_CONTEXT_DIR,
        :docs_top_k => 3,
        :enable_docs_context => true,
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
        elseif a == "--docs-context-dir" && i + 1 <= length(args)
            kwargs[:docs_context_dir] = args[i + 1]; i += 2
        elseif a == "--docs-top-k" && i + 1 <= length(args)
            kwargs[:docs_top_k] = parse(Int, args[i + 1]); i += 2
        elseif a == "--disable-docs-context"
            kwargs[:enable_docs_context] = false; i += 1
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
        docs_context_dir = kwargs[:docs_context_dir],
        docs_top_k = kwargs[:docs_top_k],
        enable_docs_context = kwargs[:enable_docs_context],
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
