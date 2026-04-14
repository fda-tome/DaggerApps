#!/usr/bin/env julia
using HTTP
using JSON3

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const PROMPTS_DIR = joinpath(REPO_ROOT, "benchmark", "prompts")
const RUNTIME_CONTEXTS_PATH = joinpath(PROMPTS_DIR, "runtime_contexts.json")
const DOCS_CONTEXT_DIR = joinpath(PROMPTS_DIR, "docs_context")
const INDEX_PATH = joinpath(DOCS_CONTEXT_DIR, "index.json")

strip_html(s::String) = replace(replace(s, r"<script[^>]*>.*?</script>"is => " "), r"<style[^>]*>.*?</style>"is => " ") |>
    x -> replace(x, r"<[^>]+>" => " ") |>
    x -> replace(x, r"\s+" => " ") |>
    strip

function safe_fetch(url::String)::String
    try
        r = HTTP.get(url; connect_timeout=20, readtimeout=40)
        r.status == 200 || return ""
        return strip_html(String(r.body))
    catch
        return ""
    end
end

function load_runtime_contexts()::Dict{String,Any}
    raw = JSON3.read(read(RUNTIME_CONTEXTS_PATH, String))
    out = Dict{String,Any}()
    for (k, v) in pairs(raw)
        d = Dict{String,Any}()
        for (kk, vv) in pairs(v)
            d[String(kk)] = vv
        end
        out[String(k)] = d
    end
    out
end

function write_json(path::String, obj)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, obj)
    end
end

function build_snippet(framework::String, title::String, source_url::String)::Dict{String,Any}
    text = safe_fetch(source_url)
    excerpt = isempty(text) ? "No fetched excerpt available. Add curated text manually for this source." : first(text, min(lastindex(text), 1200))
    Dict(
        "title" => title,
        "source_url" => source_url,
        "excerpt" => excerpt,
    )
end

function main()
    runtime_contexts = load_runtime_contexts()
    mkpath(DOCS_CONTEXT_DIR)

    index = Dict(
        "frameworks" => Dict{String,Any}()
    )

    for (framework, ctx) in runtime_contexts
        fw_dir = joinpath(DOCS_CONTEXT_DIR, framework)
        mkpath(fw_dir)
        repo_url = String(get(ctx, "repo_url", ""))
        docs_url = String(get(ctx, "docs_url", ""))

        common_paths = String[]
        if !isempty(docs_url)
            rel = joinpath(framework, "docs_primary.json")
            push!(common_paths, rel)
            write_json(joinpath(DOCS_CONTEXT_DIR, rel), build_snippet(framework, "$(framework) docs excerpt", docs_url))
        end
        if !isempty(repo_url)
            rel = joinpath(framework, "repo_primary.json")
            push!(common_paths, rel)
            write_json(joinpath(DOCS_CONTEXT_DIR, rel), build_snippet(framework, "$(framework) repo excerpt", repo_url))
        end

        index["frameworks"][framework] = Dict(
            "common" => common_paths,
            "tasks" => Dict{String,Any}(),
        )
    end

    write_json(INDEX_PATH, index)
    println("Wrote ", INDEX_PATH)
end

main()
