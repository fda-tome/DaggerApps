#!/usr/bin/env julia
# Phase 3: Statistics and plots from evaluation results. Pure post-processing; no Julia execution of samples.
# Output: figures/*.png, tables/*.md

using DataFrames
using Statistics
using JSON3
using Printf
using CairoMakie

const REPO_ROOT = dirname(@__DIR__)
const UTILS_DIR = joinpath(REPO_ROOT, "src", "utils")
const OUTPUT_DIR = joinpath(REPO_ROOT, "outputs", "evaluated")
const FIGURES_DIR = joinpath(REPO_ROOT, "figures")
const TABLES_DIR = joinpath(REPO_ROOT, "tables")

include(joinpath(UTILS_DIR, "pass_at_k.jl"))

function load_evaluated(jsonl_path::String)::DataFrame
    records = open(jsonl_path) do io
        [JSON3.read(line) for line in eachline(io) if !isempty(strip(line))]
    end
    task = String[]
    model = String[]
    framework = String[]
    sample_id = Int[]
    passed = Bool[]
    elapsed_sec = Float64[]
    for r in records
        push!(task, String(r.task))
        push!(model, String(r.model))
        push!(framework, haskey(r, :framework) ? String(r.framework) : "dagger")
        push!(sample_id, Int(r.sample_id))
        push!(passed, Bool(r.passed))
        push!(elapsed_sec, Float64(r.elapsed_sec))
    end
    DataFrame(; task, model, framework, sample_id, passed, elapsed_sec)
end

function compute_pass_at_k(df::DataFrame; k_values::Vector{Int}=[1, 5], groupby_cols::Vector{Symbol}=[:task, :model])
    g = groupby(df, groupby_cols)
    rows = []
    for sub in g
        n_total = nrow(sub)
        n_correct = sum(sub.passed)
        row = (; [c => first(sub[!, c]) for c in groupby_cols]..., n_total, n_correct)
        for k in k_values
            row = merge(row, (Symbol("pass_at_$k") => pass_at_k_unbiased(n_total, n_correct, k),))
        end
        push!(rows, row)
    end
    DataFrame(rows)
end

function format_md_cell(v)
    if v isa AbstractFloat
        return isfinite(v) ? @sprintf("%.2f", Float64(v)) : "---"
    end
    return string(v)
end

"""Save a simple figure when there is no data to plot (keeps the pipeline from crashing)."""
function save_placeholder_fig(out_path::String, message::String)
    fig = Figure(size=(720, 360))
    Label(fig[1, 1], message; fontsize=14, justification=:left)
    save(out_path, fig)
end

function write_markdown_table(df::DataFrame, out_path::String)
    cols = names(df)
    open(out_path, "w") do io
        println(io, "| " * join(string.(cols), " | ") * " |")
        println(io, "|" * join(fill("---", length(cols)), "|") * "|")
        for r in eachrow(df)
            vals = [format_md_cell(r[c]) for c in cols]
            println(io, "| " * join(vals, " | ") * " |")
        end
    end
end

function plot_pass_by_task(pass_df::DataFrame, out_path::String)
    k_cols = [n for n in names(pass_df) if startswith(string(n), "pass_at")]
    isempty(k_cols) && (save_placeholder_fig(out_path, "No pass@k columns to plot."); return)
    k1 = k_cols[1]
    tasks = unique(pass_df.task)
    models = unique(pass_df.model)
    if nrow(pass_df) == 0 || isempty(tasks) || isempty(models)
        save_placeholder_fig(out_path, "No evaluation rows — pass@k by task plot skipped.")
        return
    end
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Task", ylabel="pass@k (unbiased)", xticklabelrotation=pi/4)
    n_tasks = length(tasks)
    n_models = length(models)
    width = 0.8 / max(n_models, 1)
    for (i, model) in enumerate(models)
        ys = Float64[]
        for t in tasks
            r = pass_df[(pass_df.task .== t) .& (pass_df.model .== model), k1]
            push!(ys, isempty(r) ? NaN : r[1])
        end
        xs = (1:n_tasks) .+ (i - 0.5) * width .- 0.4
        barplot!(ax, xs, ys; width=width, label=model)
    end
    ax.xticks = (1:n_tasks, tasks)
    axislegend(ax; position=:rt)
    save(out_path, fig)
end

function compute_mean_runtime(df::DataFrame; groupby_cols::Vector{Symbol}=[:task, :framework], passed_only::Bool=true)
    sub = passed_only ? df[df.passed .== true, :] : df
    g = groupby(sub, groupby_cols)
    rows = []
    for s in g
        row = (; [c => first(s[!, c]) for c in groupby_cols]..., mean_elapsed_sec = mean(s.elapsed_sec), n_passed = nrow(s))
        push!(rows, row)
    end
    isempty(rows) && return DataFrame(task=String[], framework=String[], mean_elapsed_sec=Float64[], n_passed=Int[])
    DataFrame(rows)
end

"""Comparative runtime across frameworks. X = task, grouped bars = framework, Y = mean elapsed sec."""
function plot_runtime_comparative(runtime_df::DataFrame, out_path::String)
    if !("framework" in names(runtime_df))
        save_placeholder_fig(out_path, "No framework column — runtime plot skipped.")
        return
    end
    tasks = sort(unique(runtime_df.task))
    frameworks = sort(unique(runtime_df.framework))
    if nrow(runtime_df) == 0 || isempty(tasks) || isempty(frameworks)
        save_placeholder_fig(
            out_path,
            "No passed evaluations — mean runtime is empty (passed_only=true). Failures still appear in pass@k tables.",
        )
        return
    end
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Task", ylabel="Mean runtime (s)", xticklabelrotation=pi/4)
    n_tasks = length(tasks)
    n_fw = length(frameworks)
    width = 0.8 / max(n_fw, 1)
    for (i, fw) in enumerate(frameworks)
        ys = Float64[]
        for t in tasks
            r = runtime_df[(runtime_df.task .== t) .& (runtime_df.framework .== fw), :mean_elapsed_sec]
            push!(ys, isempty(r) ? NaN : r[1])
        end
        xs = (1:n_tasks) .+ (i - 0.5) * width .- 0.4
        barplot!(ax, xs, ys; width=width, label=fw)
    end
    ax.xticks = (1:n_tasks, tasks)
    axislegend(ax; position=:rt)
    save(out_path, fig)
end

"""Comparative pass@k across frameworks (Dagger, Iris, Legate). X = task, grouped bars = framework."""
function plot_pass_by_framework(pass_df::DataFrame, out_path::String)
    k_cols = [n for n in names(pass_df) if startswith(string(n), "pass_at")]
    if !("framework" in names(pass_df))
        save_placeholder_fig(out_path, "No framework column — comparative pass@k plot skipped.")
        return
    end
    isempty(k_cols) && (save_placeholder_fig(out_path, "No pass@k columns — comparative plot skipped."); return)
    k1 = k_cols[1]
    frameworks = sort(unique(pass_df.framework))
    tasks = sort(unique(pass_df.task))
    if nrow(pass_df) == 0 || isempty(tasks) || isempty(frameworks)
        save_placeholder_fig(out_path, "No rows for pass@k by framework — plot skipped.")
        return
    end
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Task", ylabel="pass@k (unbiased)", xticklabelrotation=pi/4)
    n_tasks = length(tasks)
    n_fw = length(frameworks)
    width = 0.8 / max(n_fw, 1)
    for (i, fw) in enumerate(frameworks)
        ys = Float64[]
        for t in tasks
            r = pass_df[(pass_df.task .== t) .& (pass_df.framework .== fw), k1]
            # Average over models if multiple
            vals = r[.!isnan.(r)]
            push!(ys, isempty(vals) ? NaN : mean(vals))
        end
        xs = (1:n_tasks) .+ (i - 0.5) * width .- 0.4
        barplot!(ax, xs, ys; width=width, label=fw)
    end
    ax.xticks = (1:n_tasks, tasks)
    axislegend(ax; position=:rt)
    save(out_path, fig)
end

function main()
    candidates = filter(f -> endswith(f, ".jsonl"), readdir(OUTPUT_DIR; join=true))
    evaluated = length(ARGS) >= 1 ? ARGS[1] : (isempty(candidates) ? "" : first(candidates))
    isempty(evaluated) && error("No evaluated JSONL found in $OUTPUT_DIR and no path given. Provide path as ARGS[1].")
    evaluated = abspath(evaluated)
    isfile(evaluated) || error("File not found: $evaluated")

    df = load_evaluated(evaluated)
    pass_df = compute_pass_at_k(df; k_values=[1, 5])
    pass_df_fw = compute_pass_at_k(df; k_values=[1, 5], groupby_cols=[:task, :model, :framework])
    runtime_df = compute_mean_runtime(df; groupby_cols=[:task, :framework], passed_only=true)
    mkpath(FIGURES_DIR)
    mkpath(TABLES_DIR)

    write_markdown_table(pass_df, joinpath(TABLES_DIR, "pass_at_k.md"))
    write_markdown_table(pass_df_fw, joinpath(TABLES_DIR, "pass_at_k_by_framework.md"))
    write_markdown_table(runtime_df, joinpath(TABLES_DIR, "runtime_by_framework.md"))
    plot_pass_by_task(pass_df, joinpath(FIGURES_DIR, "pass_at_k_by_task.png"))
    plot_pass_by_framework(pass_df_fw, joinpath(FIGURES_DIR, "pass_at_k_comparative.png"))
    plot_runtime_comparative(runtime_df, joinpath(FIGURES_DIR, "runtime_comparative.png"))

    println("Wrote ", joinpath(TABLES_DIR, "pass_at_k.md"))
    println("Wrote ", joinpath(TABLES_DIR, "pass_at_k_by_framework.md"))
    println("Wrote ", joinpath(TABLES_DIR, "runtime_by_framework.md"))
    println("Wrote ", joinpath(FIGURES_DIR, "pass_at_k_by_task.png"))
    println("Wrote ", joinpath(FIGURES_DIR, "pass_at_k_comparative.png"))
    println("Wrote ", joinpath(FIGURES_DIR, "runtime_comparative.png"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
