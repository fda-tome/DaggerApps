#!/usr/bin/env julia

using Printf

const REPO_ROOT = abspath(@__DIR__)
const APPS_DIR = joinpath(REPO_ROOT, "apps")
const BENCHMARKS_DIR = joinpath(REPO_ROOT, "benchmarks")

function _discover_apps()::Vector{String}
    if !isdir(APPS_DIR)
        error("apps/ directory not found at: $APPS_DIR")
    end
    apps = String[]
    for name in sort(readdir(APPS_DIR))
        if startswith(name, ".")
            continue
        end
        if isdir(joinpath(APPS_DIR, name))
            push!(apps, name)
        end
    end
    return apps
end

function _run_app_benchmark(app::AbstractString)
    bench_script = joinpath(BENCHMARKS_DIR, "scripts", string(app, ".jl"))
    if !isfile(bench_script)
        @warn "No benchmark entrypoint found; skipping" app bench_script
        return
    end

    app_project = joinpath(APPS_DIR, app)
    cmd = Base.julia_cmd()
    cmd = `$cmd --project=$app_project $bench_script`

    println()
    println("="^70)
    println("Running benchmark: $app")
    println("  project: $app_project")
    println("  script:  $bench_script")
    println("="^70)
    run(cmd)
end

"""
    run_all_benchmarks([apps...])

Runs all per-app benchmark entrypoints `benchmarks/scripts/<app>.jl` for each app in `apps/`,
or only the apps listed in `ARGS`.

Each benchmark is executed in a separate Julia process with `--project=apps/<app>`.
"""
function run_all_benchmarks(apps::Vector{String}=ARGS)
    all_apps = _discover_apps()
    selected = isempty(apps) ? all_apps : apps

    unknown = setdiff(selected, all_apps)
    if !isempty(unknown)
        error("Unknown app(s): $(join(unknown, ", ")). Known: $(join(all_apps, ", "))")
    end

    for app in selected
        _run_app_benchmark(app)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_benchmarks()
end
