module DaggerAppsBenchmarks

export run_seam_carving, load_seam_carving, run_game_of_life, load_game_of_life

const _seam_loaded = Ref(false)
const _seam_run = Ref{Union{Nothing, Function}}(nothing)
const _seam_mod = Ref{Union{Nothing, Module}}(nothing)
const _life_loaded = Ref(false)
const _life_run = Ref{Union{Nothing, Function}}(nothing)
const _life_mod = Ref{Union{Nothing, Module}}(nothing)

function _load_runner(path::AbstractString, modname::Symbol)
    mod = if isdefined(@__MODULE__, modname)
        getfield(@__MODULE__, modname)
    else
        Core.eval(@__MODULE__, :(module $modname end))
    end
    Base.include(mod, path)
    runner = Core.eval(mod, :run_benchmark)
    return mod, runner
end

function load_seam_carving()
    if !_seam_loaded[]
        path = joinpath(@__DIR__, "..", "seam-carving.jl")
        mod, runner = _load_runner(path, :SeamCarvingBench)
        _seam_mod[] = mod
        _seam_run[] = runner
        _seam_loaded[] = true
    end
    return nothing
end

function run_seam_carving(; kwargs...)
    load_seam_carving()
    runner = _seam_run[]
    runner === nothing && error("run_benchmark was not loaded for seam-carving.")
    return Base.invokelatest(runner; kwargs...)
end

function load_game_of_life()
    if !_life_loaded[]
        path = joinpath(@__DIR__, "..", "game-of-life.jl")
        mod, runner = _load_runner(path, :GameOfLifeBench)
        _life_mod[] = mod
        _life_run[] = runner
        _life_loaded[] = true
    end
    return nothing
end

function run_game_of_life(; kwargs...)
    load_game_of_life()
    runner = _life_run[]
    runner === nothing && error("run_benchmark was not loaded for game-of-life.")
    return Base.invokelatest(runner; kwargs...)
end

end
