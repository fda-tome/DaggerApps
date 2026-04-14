#!/usr/bin/env julia
using Statistics

# Parse NDJSON from SEAM_PERF_LOG (westrick_timings lines) into a scaling table.
#
# Usage:
#   julia summarize-westrick-perf-log.jl [LOG_PATH] [--drop-first] [--csv]
#   julia summarize-westrick-perf-log.jl [LOG_PATH] --drop-first --check-monotonic [--monotonic-rel-tol=0.001]
#
# Default LOG_PATH: <paper>/.cursor/debug-d35653.log (three levels up from this script).
#
# --drop-first: drop the first timing sample in each contiguous block with the same nthreads
#   (reduces JIT/first-carve bias within each Julia subprocess).

const RX_NT = r"\"nthreads\":([0-9]+)"
const RX_TOT = r"\"total_s\":([0-9.eE+-]+)"
const RX_DP = r"\"westrick_dp_s\":([0-9.eE+-]+)"
const RX_FR = r"\"dp_frac\":([0-9.eE+-]+)"

function default_log_path()::String
    script = @__DIR__
    paper = abspath(joinpath(script, "..", "..", ".."))
    return joinpath(paper, ".cursor", "debug-d35653.log")
end

function parse_timings_line(line::String)::Union{Nothing,NamedTuple}
    occursin("westrick_timings", line) || return nothing
    mnt = match(RX_NT, line)
    mtot = match(RX_TOT, line)
    (mnt === nothing || mtot === nothing) && return nothing
    nthreads = parse(Int, mnt.captures[1])
    total_s = parse(Float64, mtot.captures[1])
    mdp = match(RX_DP, line)
    westrick_dp_s = mdp === nothing ? NaN : parse(Float64, mdp.captures[1])
    mfr = match(RX_FR, line)
    dp_frac = mfr === nothing ? NaN : parse(Float64, mfr.captures[1])
    return (; nthreads, total_s, westrick_dp_s, dp_frac)
end

function contiguous_blocks(rows::Vector{<:NamedTuple})
    blocks = Vector{typeof(rows)}()
    isempty(rows) && return blocks
    cur = [rows[1]]
    for i in 2:length(rows)
        if rows[i].nthreads == rows[i - 1].nthreads
            push!(cur, rows[i])
        else
            push!(blocks, cur)
            cur = [rows[i]]
        end
    end
    push!(blocks, cur)
    return blocks
end

function parse_monotonic_rel_tol(args::Vector{String})::Float64
    for a in args
        if startswith(a, "--monotonic-rel-tol=")
            return parse(Float64, a[(length("--monotonic-rel-tol=") + 1):end])
        end
    end
    return 0.0
end

"""Require median wall time to strictly decrease (within `rel_tol`) as `nthreads` increases row-by-row."""
function check_median_monotonic(summaries; rel_tol::Float64 = 0.0)::Bool
    length(summaries) < 2 && return true
    srt = sort(collect(summaries), by = x -> x.nthreads)
    bad = false
    for i in 1:(length(srt) - 1)
        a, b = srt[i], srt[i + 1]
        b.nthreads <= a.nthreads && continue
        # More threads ⇒ should be faster: lower median time (allow tiny measurement wobble via rel_tol)
        lim = a.median_total_s * (1 + rel_tol)
        if b.median_total_s > lim
            println(stderr, "MONOTONIC_FAIL  threads $(a.nthreads) median_s=$(a.median_total_s)  ->  $(b.nthreads) median_s=$(b.median_total_s)  (not strict decrease; rel_tol=$(rel_tol))")
            bad = true
        end
    end
    return !bad
end

function main()
    args = ARGS
    log_path = default_log_path()
    if !isempty(args) && !startswith(args[1], "-")
        log_path = args[1]
        args = args[2:end]
    end
    drop_first = "--drop-first" in args
    csv = "--csv" in args
    check_mono = "--check-monotonic" in args
    rel_tol = parse_monotonic_rel_tol(args)

    isfile(log_path) || error("Log not found: $log_path")

    rows = NamedTuple[]
    for line in eachline(log_path)
        r = parse_timings_line(line)
        r === nothing || push!(rows, r)
    end
    isempty(rows) && error("No westrick_timings lines in $log_path")

    summaries = NamedTuple[]
    for block in contiguous_blocks(rows)
        svec = drop_first && length(block) > 1 ? block[2:end] : block
        isempty(svec) && continue
        t = svec[1].nthreads
        totals = [x.total_s for x in svec]
        μtot = mean(totals)
        medtot = median(totals)
        dps = [x.westrick_dp_s for x in svec if !isnan(x.westrick_dp_s)]
        frs = [x.dp_frac for x in svec if !isnan(x.dp_frac)]
        μdp = isempty(dps) ? NaN : mean(dps)
        μfr = isempty(frs) ? NaN : mean(frs)
        push!(summaries, (; nthreads=t, n=length(svec), mean_total_s=μtot, median_total_s=medtot,
            mean_westrick_dp_s=μdp, mean_dp_frac=μfr))
    end

    if csv
        println("nthreads,n,mean_total_s,median_total_s,mean_westrick_dp_s,mean_dp_frac")
        for s in summaries
            println("$(s.nthreads),$(s.n),$(s.mean_total_s),$(s.median_total_s),$(s.mean_westrick_dp_s),$(s.mean_dp_frac)")
        end
    else
        println(rpad("threads", 8), rpad("n", 4), rpad("mean_tot_s", 12), rpad("med_tot_s", 12),
            rpad("mean_dp_s", 12), rpad("mean_dp%", 10))
        for s in summaries
            println(rpad(s.nthreads, 8), rpad(s.n, 4), rpad(string(round(s.mean_total_s; digits=4)), 12),
                rpad(string(round(s.median_total_s; digits=4)), 12),
                rpad(string(round(s.mean_westrick_dp_s; digits=4)), 12),
                rpad(string(round(s.mean_dp_frac * 100; digits=2)), 10))
        end
    end
    if check_mono
        ok = check_median_monotonic(summaries; rel_tol = rel_tol)
        if ok
            println(stderr, "MONOTONIC_OK  median_total_s decreases at every step (nthreads ascending, rel_tol=$(rel_tol))")
        else
            println(stderr, "MONOTONIC_CHECK_FAILED  see MONOTONIC_FAIL lines above; try larger SEAM_ROWS, SEAM_WESTRICK_BLOCK_WIDTH_AUTO_TASK_TARGET, or BENCH_RUNS")
            exit(1)
        end
    end
    return nothing
end

main()
