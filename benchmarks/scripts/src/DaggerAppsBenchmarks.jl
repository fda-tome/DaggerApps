module DaggerAppsBenchmarks

export run_seam_carving, load_seam_carving, run_game_of_life, load_game_of_life, run_heat_propagation, load_heat_propagation
export run_barnes_hut, load_barnes_hut
export run_game_of_life_gpu_size_sweep, run_heat_propagation_gpu_size_sweep

const _seam_loaded = Ref(false)
const _seam_run = Ref{Union{Nothing, Function}}(nothing)
const _seam_mod = Ref{Union{Nothing, Module}}(nothing)
const _life_loaded = Ref(false)
const _life_run = Ref{Union{Nothing, Function}}(nothing)
const _life_mod = Ref{Union{Nothing, Module}}(nothing)
const _heat_loaded = Ref(false)
const _heat_run = Ref{Union{Nothing, Function}}(nothing)
const _heat_mod = Ref{Union{Nothing, Module}}(nothing)
const _barnes_loaded = Ref(false)
const _barnes_run = Ref{Union{Nothing, Function}}(nothing)
const _barnes_mod = Ref{Union{Nothing, Module}}(nothing)

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

function run_game_of_life_gpu_size_sweep(; kwargs...)
    load_game_of_life()
    mod = _life_mod[]
    mod === nothing && error("Game-of-life benchmark module was not loaded.")
    runner = Core.eval(mod, :run_gpu_size_sweep)
    return Base.invokelatest(runner; kwargs...)
end

function load_heat_propagation()
    if !_heat_loaded[]
        path = joinpath(@__DIR__, "..", "heat-propagation.jl")
        mod, runner = _load_runner(path, :HeatPropagationBench)
        _heat_mod[] = mod
        _heat_run[] = runner
        _heat_loaded[] = true
    end
    return nothing
end

function run_heat_propagation(; kwargs...)
    load_heat_propagation()
    runner = _heat_run[]
    runner === nothing && error("run_benchmark was not loaded for heat-propagation.")
    return Base.invokelatest(runner; kwargs...)
end

function run_heat_propagation_gpu_size_sweep(; kwargs...)
    load_heat_propagation()
    mod = _heat_mod[]
    mod === nothing && error("Heat-propagation benchmark module was not loaded.")
    runner = Core.eval(mod, :run_gpu_size_sweep)
    return Base.invokelatest(runner; kwargs...)
end

function load_barnes_hut()
    if !_barnes_loaded[]
        path = joinpath(@__DIR__, "..", "barnes-hut.jl")
        mod, runner = _load_runner(path, :BarnesHutBench)
        _barnes_mod[] = mod
        _barnes_run[] = runner
        _barnes_loaded[] = true
    end
    return nothing
end

function run_barnes_hut(; kwargs...)
    load_barnes_hut()
    runner = _barnes_run[]
    runner === nothing && error("run_benchmark was not loaded for barnes-hut.")
    return Base.invokelatest(runner; kwargs...)
end

end
