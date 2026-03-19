# Framework-specific runners: Iris (C++), Legate (Python), PaRSEC (C).
# Contract: (task_name, task_file, code, response_raw; timeout_sec) -> (passed, results, elapsed, error, stderr)
# Pass/fail convention: last non-empty line of stdout is "PASS" or "FAIL"; exit code 0 for pass.

function run_with_timeout_capture(cmd, timeout_sec::Real)
    out_io = IOBuffer()
    err_io = IOBuffer()
    proc = run(pipeline(cmd, stdout=out_io, stderr=err_io), wait=false)
    start = time()
    timed_out = false
    while process_running(proc)
        if time() - start >= timeout_sec
            kill(proc)
            timed_out = true
            break
        end
        sleep(0.1)
    end
    elapsed = time() - start
    sleep(0.2)
    out_str = String(take!(out_io))
    err_str = String(take!(err_io))
    exit_ok = !timed_out && process_exited(proc) && something(proc.exitcode, -1) == 0
    (exit_ok, out_str, err_str, elapsed)
end

function parse_pass_fail(stdout_str::String, exit_ok::Bool)
    lines = [strip(l) for l in split(stdout_str, '\n') if !isempty(strip(l))]
    last_line = isempty(lines) ? "" : lines[end]
    passed = exit_ok && (occursin(r"^PASS$", last_line) || last_line == "PASS")
    (passed, [passed])
end

"""Iris runner: C++. Build with g++ (or IRIS_CC env), run binary. Override via IRIS_BUILD_CMD / IRIS_RUN_CMD env."""
function run_iris(task_name::String, task_file::String, code::String, response_raw::String; timeout_sec::Real=120.0)
    dir = mktempdir()
    try
        src = endswith(task_file, ".cpp") ? task_file : "main.cpp"
        src_path = joinpath(dir, src)
        write(src_path, code)
        build_cmd = get(ENV, "IRIS_BUILD_CMD", "g++ -o main " * src)
        build_success, _, build_err, _ = run_with_timeout_capture(Cmd(`sh -c $build_cmd`, dir=dir), min(60.0, timeout_sec))
        if !build_success
            return (passed=false, results=Bool[], elapsed=0.0, error="build failed: " * build_err, stderr=build_err)
        end
        run_cmd = get(ENV, "IRIS_RUN_CMD", "./main")
        exit_ok, out_str, err_str, elapsed = run_with_timeout_capture(Cmd(`sh -c $run_cmd`, dir=dir), timeout_sec)
        passed, results = parse_pass_fail(out_str, exit_ok)
        return (passed=passed, results=results, elapsed=elapsed, error=passed ? nothing : "run failed or did not print PASS", stderr=err_str)
    finally
        rm(dir; force=true, recursive=true)
    end
end

"""Legate runner: Python. Run with python (or LEGATE_PYTHON env). Override via LEGATE_RUN_CMD env."""
function run_legate(task_name::String, task_file::String, code::String, response_raw::String; timeout_sec::Real=120.0)
    dir = mktempdir()
    try
        script = endswith(task_file, ".py") ? task_file : "solution.py"
        script_path = joinpath(dir, script)
        write(script_path, code)
        run_cmd = get(ENV, "LEGATE_RUN_CMD", "python " * script)
        exit_ok, out_str, err_str, elapsed = run_with_timeout_capture(Cmd(`sh -c $run_cmd`, dir=dir), timeout_sec)
        passed, results = parse_pass_fail(out_str, exit_ok)
        return (passed=passed, results=results, elapsed=elapsed, error=passed ? nothing : "run failed or did not print PASS", stderr=err_str)
    finally
        rm(dir; force=true, recursive=true)
    end
end

"""PaRSEC runner: C. Build with gcc (or PARSEC_CC env), run binary. Override via PARSEC_BUILD_CMD / PARSEC_RUN_CMD env."""
function run_parsec(task_name::String, task_file::String, code::String, response_raw::String; timeout_sec::Real=120.0)
    dir = mktempdir()
    try
        src = endswith(task_file, ".c") ? task_file : "main.c"
        src_path = joinpath(dir, src)
        write(src_path, code)
        build_cmd = get(ENV, "PARSEC_BUILD_CMD", "gcc -o main " * src)
        build_success, _, build_err, _ = run_with_timeout_capture(Cmd(`sh -c $build_cmd`, dir=dir), min(60.0, timeout_sec))
        if !build_success
            return (passed=false, results=Bool[], elapsed=0.0, error="build failed: " * build_err, stderr=build_err)
        end
        run_cmd = get(ENV, "PARSEC_RUN_CMD", "./main")
        exit_ok, out_str, err_str, elapsed = run_with_timeout_capture(Cmd(`sh -c $run_cmd`, dir=dir), timeout_sec)
        passed, results = parse_pass_fail(out_str, exit_ok)
        return (passed=passed, results=results, elapsed=elapsed, error=passed ? nothing : "run failed or did not print PASS", stderr=err_str)
    finally
        rm(dir; force=true, recursive=true)
    end
end
