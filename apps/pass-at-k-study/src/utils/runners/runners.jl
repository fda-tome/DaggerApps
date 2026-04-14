# Framework-specific runners: Iris (C++), Legate (Python).
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

"""Expand `{SRC}` / `{SCRIPT}` in env-supplied commands (see `docs/RUNNER_ENV.md`)."""
function expand_runner_cmd(template::String, src::String, script::String=src)
    replace(replace(template, "{SRC}" => src), "{SCRIPT}" => script)
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
        env_iris = get(ENV, "IRIS_BUILD_CMD", "")
        build_cmd = if isempty(strip(env_iris))
            "g++ -o main " * src
        elseif occursin("{SRC}", env_iris)
            expand_runner_cmd(env_iris, src)
        else
            env_iris * " " * src
        end
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

"""Legate runner: Python. Set `LEGATE_RUN_CMD` or `LEGATE_PYTHON` (see docs/RUNNER_ENV.md). Default: `python3 <script>` (portable refs use NumPy only; conda Legate → set `LEGATE_PYTHON`)."""
function run_legate(task_name::String, task_file::String, code::String, response_raw::String; timeout_sec::Real=120.0)
    dir = mktempdir()
    try
        script = endswith(task_file, ".py") ? task_file : "solution.py"
        script_path = joinpath(dir, script)
        write(script_path, code)
        default_py = get(ENV, "LEGATE_PYTHON", "python3")
        env_leg = get(ENV, "LEGATE_RUN_CMD", "")
        run_cmd = if isempty(strip(env_leg))
            default_py * " " * script
        else
            expand_runner_cmd(env_leg, script, script)
        end
        exit_ok, out_str, err_str, elapsed = run_with_timeout_capture(Cmd(`sh -c $run_cmd`, dir=dir), timeout_sec)
        passed, results = parse_pass_fail(out_str, exit_ok)
        return (passed=passed, results=results, elapsed=elapsed, error=passed ? nothing : "run failed or did not print PASS", stderr=err_str)
    finally
        rm(dir; force=true, recursive=true)
    end
end
