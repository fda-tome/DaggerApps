# Safe evaluation with timeout and resource limits.
# Run generated code in a subprocess or with a timeout; used by evaluate.jl.

"""
    run_with_timeout(cmd::Cmd, timeout_sec::Real; stdout_ref, stderr_ref)

Run `cmd` with a timeout. On timeout, kill the process.
stdout_ref and stderr_ref can be Ref{String} or similar to capture output.
Returns (success::Bool, exit_code::Int, elapsed::Float64).
"""
function run_with_timeout(cmd::Cmd, timeout_sec::Real;
                          stdout_ref=nothing, stderr_ref=nothing)
    start = time()
    proc = run(pipeline(cmd, stdout=stdout_ref !== nothing ? stdout_ref[] : stdout,
                        stderr=stderr_ref !== nothing ? stderr_ref[] : stderr),
               wait=false)
    while process_running(proc)
        elapsed = time() - start
        if elapsed >= timeout_sec
            kill(proc)
            return false, -1, elapsed
        end
        sleep(0.1)
    end
    elapsed = time() - start
    success = proc.exitcode == 0
    (success, something(proc.exitcode, -1), elapsed)
end

"""
    run_julia_script_with_timeout(script_path::String, timeout_sec::Real=120;
                                 project_path=nothing) -> (success, exit_code, elapsed, stdout_str, stderr_str)

Run a Julia script with the given timeout. Uses the project at project_path if given.
"""
function run_julia_script_with_timeout(script_path::String, timeout_sec::Real=120;
                                      project_path=nothing)
    project_path = something(project_path, pwd())
    out_io = IOBuffer()
    err_io = IOBuffer()
    jl_cmd = pipeline(`julia --project=$project_path $script_path`, stdout=out_io, stderr=err_io)
    start = time()
    proc = run(jl_cmd, wait=false)
    while process_running(proc)
        (time() - start >= timeout_sec) && (kill(proc); break)
        sleep(0.1)
    end
    elapsed = time() - start
    success = process_exited(proc) && proc.exitcode == 0
    exit_code = something(proc.exitcode, -1)
    stdout_str = String(take!(out_io))
    stderr_str = String(take!(err_io))
    (success, exit_code, elapsed, stdout_str, stderr_str)
end
