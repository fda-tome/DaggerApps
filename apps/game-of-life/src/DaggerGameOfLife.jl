module DaggerGameOfLife

using Dagger
import Dagger: @stencil, Wrap, Pad

@inline function _normalize_boundary(boundary, pad_value::Bool)
    if boundary isa Wrap || boundary isa Pad
        return boundary
    elseif boundary isa Symbol
        boundary === :wrap && return Wrap()
        boundary === :pad && return Pad(pad_value)
    elseif boundary isa AbstractString
        s = lowercase(strip(boundary))
        s == "wrap" && return Wrap()
        s == "pad" && return Pad(pad_value)
    end
    throw(ArgumentError("Unsupported boundary=$boundary. Use :wrap, :pad, Wrap(), or Pad(...)."))
end

@inline function _normalize_serial_boundary(boundary, pad_value::Bool)
    if boundary isa Symbol
        boundary === :wrap && return (:wrap, pad_value)
        boundary === :pad && return (:pad, pad_value)
    elseif boundary isa Wrap
        return (:wrap, pad_value)
    elseif boundary isa Pad
        return (:pad, Bool(getfield(boundary, :padval)))
    elseif boundary isa AbstractString
        s = lowercase(strip(boundary))
        s == "wrap" && return (:wrap, pad_value)
        s == "pad" && return (:pad, pad_value)
    end
    throw(ArgumentError("Unsupported boundary=$boundary. Use :wrap, :pad, Wrap(), or Pad(...)."))
end

@inline _survives(alive::Bool, neighbors::Int) = alive ? (neighbors == 2 || neighbors == 3) : (neighbors == 3)

function random_world(rows::Integer, cols::Integer; density::Real=0.25)
    rows > 0 || throw(ArgumentError("rows must be > 0"))
    cols > 0 || throw(ArgumentError("cols must be > 0"))
    0.0 <= density <= 1.0 || throw(ArgumentError("density must be in [0, 1]"))
    return rand(rows, cols) .< density
end

function seed_glider!(world::AbstractMatrix{Bool}; row::Int=2, col::Int=2)
    H, W = size(world)
    row + 2 <= H || throw(ArgumentError("glider does not fit rows: row=$(row), H=$(H)"))
    col + 2 <= W || throw(ArgumentError("glider does not fit cols: col=$(col), W=$(W)"))

    world[row + 0, col + 1] = true
    world[row + 1, col + 2] = true
    world[row + 2, col + 0] = true
    world[row + 2, col + 1] = true
    world[row + 2, col + 2] = true
    return world
end

function seed_blinker!(world::AbstractMatrix{Bool}; row::Int=2, col::Int=2, horizontal::Bool=true)
    H, W = size(world)
    if horizontal
        row <= H || throw(ArgumentError("row out of bounds"))
        col + 2 <= W || throw(ArgumentError("horizontal blinker does not fit cols"))
        world[row, col + 0] = true
        world[row, col + 1] = true
        world[row, col + 2] = true
    else
        row + 2 <= H || throw(ArgumentError("vertical blinker does not fit rows"))
        col <= W || throw(ArgumentError("col out of bounds"))
        world[row + 0, col] = true
        world[row + 1, col] = true
        world[row + 2, col] = true
    end
    return world
end

function _step_cpu_serial!(dst::AbstractMatrix{Bool}, src::AbstractMatrix{Bool}; boundary::Symbol=:wrap, pad_value::Bool=false)
    H, W = size(src)
    @inbounds for y in 1:H
        for x in 1:W
            neigh = 0
            for dy in -1:1
                for dx in -1:1
                    if dy == 0 && dx == 0
                        continue
                    end
                    yy = y + dy
                    xx = x + dx
                    if boundary === :wrap
                        neigh += Int(src[mod1(yy, H), mod1(xx, W)])
                    elseif boundary === :pad
                        if (1 <= yy <= H) && (1 <= xx <= W)
                            neigh += Int(src[yy, xx])
                        else
                            neigh += Int(pad_value)
                        end
                    else
                        throw(ArgumentError("Unsupported boundary symbol=$boundary. Use :wrap or :pad."))
                    end
                end
            end
            dst[y, x] = _survives(src[y, x], neigh)
        end
    end
    return dst
end

function game_of_life_cpu_serial(initial::AbstractMatrix{Bool}; steps::Int=1, boundary=:wrap, pad_value::Bool=false)
    steps >= 0 || throw(ArgumentError("steps must be >= 0"))
    serial_boundary, serial_pad = _normalize_serial_boundary(boundary, pad_value)
    cur = copy(initial)
    nxt = similar(cur)
    for _ in 1:steps
        _step_cpu_serial!(nxt, cur; boundary=serial_boundary, pad_value=serial_pad)
        cur, nxt = nxt, cur
    end
    return cur
end

function _step_stencil!(dst::Dagger.DArray{Bool,2}, src::Dagger.DArray{Bool,2}, boundary_obj)
    Dagger.spawn_datadeps() do
        @stencil begin
            dst[idx] = begin
                nhood = @neighbors(src[idx], 1, boundary_obj)
                neigh = sum(nhood) - Int(src[idx])
                _survives(src[idx], neigh)
            end
        end
    end
    return dst
end

function _run_stencil(tiles::Dagger.DArray{Bool,2}; steps::Int=1, boundary=:wrap, pad_value::Bool=false, return_darray::Bool=false)
    steps >= 0 || throw(ArgumentError("steps must be >= 0"))
    boundary_obj = _normalize_boundary(boundary, pad_value)

    cur = tiles
    nxt = similar(cur, Bool)
    for _ in 1:steps
        _step_stencil!(nxt, cur, boundary_obj)
        cur, nxt = nxt, cur
    end

    return return_darray ? cur : collect(cur)
end

function game_of_life_dagger_stencil(initial::AbstractMatrix{Bool};
    steps::Int=1,
    block_h::Int=64,
    block_w::Int=64,
    boundary=:wrap,
    pad_value::Bool=false,
    return_darray::Bool=false,
)
    block_h > 0 || throw(ArgumentError("block_h must be > 0"))
    block_w > 0 || throw(ArgumentError("block_w must be > 0"))
    tiles = DArray(copy(initial), Blocks(block_h, block_w))
    return _run_stencil(tiles; steps=steps, boundary=boundary, pad_value=pad_value, return_darray=return_darray)
end

function game_of_life_dagger_stencil(initial::Dagger.DArray{Bool,2};
    steps::Int=1,
    boundary=:wrap,
    pad_value::Bool=false,
    return_darray::Bool=false,
)
    return _run_stencil(initial; steps=steps, boundary=boundary, pad_value=pad_value, return_darray=return_darray)
end

alive_count(world::AbstractMatrix{Bool}) = count(identity, world)
alive_count(world::Dagger.DArray{Bool,2}) = count(identity, collect(world))

function run_pipeline(initial::AbstractMatrix{Bool};
    steps::Int=10,
    block_h::Int=64,
    block_w::Int=64,
    boundary=:wrap,
    pad_value::Bool=false,
)
    serial = game_of_life_cpu_serial(initial; steps=steps, boundary=boundary, pad_value=pad_value)
    dagger = game_of_life_dagger_stencil(initial;
        steps=steps,
        block_h=block_h,
        block_w=block_w,
        boundary=boundary,
        pad_value=pad_value,
        return_darray=false,
    )
    return serial, dagger
end

end
