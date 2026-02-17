module DaggerHeatPropagation

using Dagger
using Plots
import Dagger: @stencil, Pad, Wrap

@inline function _normalize_alpha(alpha::Real)
    a = Float32(alpha)
    0f0 < a <= 0.25f0 || throw(ArgumentError("alpha must be in (0, 0.25] for explicit 2D diffusion stability; got $alpha"))
    return a
end

@inline function _normalize_boundary(boundary, pad_value::Float32)
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

@inline function _normalize_serial_boundary(boundary, pad_value::Float32)
    if boundary isa Symbol
        boundary === :wrap && return (:wrap, pad_value)
        boundary === :pad && return (:pad, pad_value)
    elseif boundary isa Wrap
        return (:wrap, pad_value)
    elseif boundary isa Pad
        return (:pad, Float32(getfield(boundary, :padval)))
    elseif boundary isa AbstractString
        s = lowercase(strip(boundary))
        s == "wrap" && return (:wrap, pad_value)
        s == "pad" && return (:pad, pad_value)
    end
    throw(ArgumentError("Unsupported boundary=$boundary. Use :wrap, :pad, Wrap(), or Pad(...)."))
end

function ambient_plate(rows::Integer, cols::Integer; ambient::Real=0.0f0)
    rows > 0 || throw(ArgumentError("rows must be > 0"))
    cols > 0 || throw(ArgumentError("cols must be > 0"))
    return fill(Float32(ambient), rows, cols)
end

function add_hotspot!(plate::AbstractMatrix{<:Real};
    row::Int=cld(size(plate, 1), 2),
    col::Int=cld(size(plate, 2), 2),
    radius::Int=6,
    temperature::Real=1.0f0,
)
    H, W = size(plate)
    1 <= row <= H || throw(ArgumentError("row out of bounds"))
    1 <= col <= W || throw(ArgumentError("col out of bounds"))
    radius >= 0 || throw(ArgumentError("radius must be >= 0"))

    temp = Float32(temperature)
    r2 = radius * radius
    @inbounds for r in max(1, row - radius):min(H, row + radius)
        for c in max(1, col - radius):min(W, col + radius)
            dr = r - row
            dc = c - col
            if dr * dr + dc * dc <= r2
                plate[r, c] = temp
            end
        end
    end
    return plate
end

function add_gaussian_hotspot!(plate::AbstractMatrix{<:Real};
    row::Int=cld(size(plate, 1), 2),
    col::Int=cld(size(plate, 2), 2),
    sigma::Real=10.0,
    amplitude::Real=1.0,
)
    H, W = size(plate)
    1 <= row <= H || throw(ArgumentError("row out of bounds"))
    1 <= col <= W || throw(ArgumentError("col out of bounds"))
    sigma > 0 || throw(ArgumentError("sigma must be > 0"))

    amp = Float32(amplitude)
    inv_two_sigma2 = Float32(1 / (2 * sigma * sigma))
    @inbounds for r in 1:H
        for c in 1:W
            dr = r - row
            dc = c - col
            plate[r, c] += amp * exp(-(dr * dr + dc * dc) * inv_two_sigma2)
        end
    end
    return plate
end

@inline function _sample(src::AbstractMatrix{Float32}, r::Int, c::Int, H::Int, W::Int, boundary::Symbol, pad_value::Float32)
    if boundary === :wrap
        return src[mod1(r, H), mod1(c, W)]
    elseif boundary === :pad
        return (1 <= r <= H && 1 <= c <= W) ? src[r, c] : pad_value
    else
        throw(ArgumentError("Unsupported boundary symbol=$boundary. Use :wrap or :pad."))
    end
end

function _step_cpu_serial!(dst::AbstractMatrix{Float32}, src::AbstractMatrix{Float32}; alpha::Float32, boundary::Symbol=:pad, pad_value::Float32=0f0)
    H, W = size(src)
    @inbounds for r in 1:H
        for c in 1:W
            center = src[r, c]
            north = _sample(src, r - 1, c, H, W, boundary, pad_value)
            south = _sample(src, r + 1, c, H, W, boundary, pad_value)
            west = _sample(src, r, c - 1, H, W, boundary, pad_value)
            east = _sample(src, r, c + 1, H, W, boundary, pad_value)
            dst[r, c] = center + alpha * (north + south + west + east - 4f0 * center)
        end
    end
    return dst
end

function heat_propagate_cpu_serial(initial::AbstractMatrix{<:Real};
    steps::Int=200,
    alpha::Real=0.2,
    boundary=:pad,
    pad_value::Real=0.0,
)
    steps >= 0 || throw(ArgumentError("steps must be >= 0"))
    a = _normalize_alpha(alpha)
    serial_boundary, serial_pad = _normalize_serial_boundary(boundary, Float32(pad_value))

    cur = Float32.(initial)
    nxt = similar(cur)
    for _ in 1:steps
        _step_cpu_serial!(nxt, cur; alpha=a, boundary=serial_boundary, pad_value=serial_pad)
        cur, nxt = nxt, cur
    end
    return cur
end

function _step_stencil!(dst::Dagger.DArray{Float32,2}, src::Dagger.DArray{Float32,2}, alpha::Float32, boundary_obj)
    Dagger.spawn_datadeps() do
        @stencil begin
            dst[idx] = begin
                nhood = @neighbors(src[idx], 1, boundary_obj)
                center = nhood[2, 2]
                north = nhood[1, 2]
                south = nhood[3, 2]
                west = nhood[2, 1]
                east = nhood[2, 3]
                center + alpha * (north + south + west + east - 4f0 * center)
            end
        end
    end
    return dst
end

function _run_stencil(tiles::Dagger.DArray{Float32,2};
    steps::Int,
    alpha::Float32,
    boundary,
    pad_value::Float32,
    return_darray::Bool,
)
    boundary_obj = _normalize_boundary(boundary, pad_value)
    cur = tiles
    nxt = similar(cur, Float32)
    for _ in 1:steps
        _step_stencil!(nxt, cur, alpha, boundary_obj)
        cur, nxt = nxt, cur
    end
    return return_darray ? cur : collect(cur)
end

function _run_stencil_history(tiles::Dagger.DArray{Float32,2};
    steps::Int,
    alpha::Float32,
    boundary,
    pad_value::Float32,
    snapshot_every::Int,
)
    snapshot_every > 0 || throw(ArgumentError("snapshot_every must be > 0"))

    boundary_obj = _normalize_boundary(boundary, pad_value)
    cur = tiles
    nxt = similar(cur, Float32)

    frames = Matrix{Float32}[]
    frame_steps = Int[]
    push!(frames, collect(cur))
    push!(frame_steps, 0)

    for step in 1:steps
        _step_stencil!(nxt, cur, alpha, boundary_obj)
        cur, nxt = nxt, cur

        if step % snapshot_every == 0 || step == steps
            push!(frames, collect(cur))
            push!(frame_steps, step)
        end
    end

    return frames, frame_steps
end

function heat_propagate_dagger_stencil(initial::AbstractMatrix{<:Real};
    steps::Int=200,
    alpha::Real=0.2,
    block_h::Int=64,
    block_w::Int=64,
    boundary=:pad,
    pad_value::Real=0.0,
    return_darray::Bool=false,
)
    steps >= 0 || throw(ArgumentError("steps must be >= 0"))
    block_h > 0 || throw(ArgumentError("block_h must be > 0"))
    block_w > 0 || throw(ArgumentError("block_w must be > 0"))

    a = _normalize_alpha(alpha)
    tiles = DArray(Float32.(initial), Blocks(block_h, block_w))
    return _run_stencil(tiles;
        steps=steps,
        alpha=a,
        boundary=boundary,
        pad_value=Float32(pad_value),
        return_darray=return_darray,
    )
end

function heat_propagate_dagger_stencil(initial::Dagger.DArray{Float32,2};
    steps::Int=200,
    alpha::Real=0.2,
    boundary=:pad,
    pad_value::Real=0.0,
    return_darray::Bool=false,
)
    steps >= 0 || throw(ArgumentError("steps must be >= 0"))
    a = _normalize_alpha(alpha)
    return _run_stencil(initial;
        steps=steps,
        alpha=a,
        boundary=boundary,
        pad_value=Float32(pad_value),
        return_darray=return_darray,
    )
end

function heat_propagate_dagger_history(initial::AbstractMatrix{<:Real};
    steps::Int=400,
    alpha::Real=0.2,
    block_h::Int=64,
    block_w::Int=64,
    boundary=:pad,
    pad_value::Real=0.0,
    snapshot_every::Int=1,
)
    steps >= 0 || throw(ArgumentError("steps must be >= 0"))
    block_h > 0 || throw(ArgumentError("block_h must be > 0"))
    block_w > 0 || throw(ArgumentError("block_w must be > 0"))

    a = _normalize_alpha(alpha)
    tiles = DArray(Float32.(initial), Blocks(block_h, block_w))
    return _run_stencil_history(tiles;
        steps=steps,
        alpha=a,
        boundary=boundary,
        pad_value=Float32(pad_value),
        snapshot_every=snapshot_every,
    )
end

function save_heat_animation(path::AbstractString;
    rows::Int=256,
    cols::Int=256,
    steps::Int=400,
    alpha::Real=0.2,
    block_h::Int=64,
    block_w::Int=64,
    ambient::Real=0.0,
    hotspot_temperature::Real=1.0,
    hotspot_radius::Int=8,
    boundary=:pad,
    pad_value=nothing,
    snapshot_every::Int=4,
    fps::Int=20,
    color=:inferno,
)
    rows > 0 || throw(ArgumentError("rows must be > 0"))
    cols > 0 || throw(ArgumentError("cols must be > 0"))
    fps > 0 || throw(ArgumentError("fps must be > 0"))

    initial = ambient_plate(rows, cols; ambient=ambient)
    add_hotspot!(initial;
        row=cld(rows, 2),
        col=cld(cols, 2),
        radius=hotspot_radius,
        temperature=hotspot_temperature,
    )

    effective_pad = isnothing(pad_value) ? Float32(ambient) : Float32(pad_value)
    frames, frame_steps = heat_propagate_dagger_history(initial;
        steps=steps,
        alpha=alpha,
        block_h=block_h,
        block_w=block_w,
        boundary=boundary,
        pad_value=effective_pad,
        snapshot_every=snapshot_every,
    )

    min_t = minimum(frames[1])
    max_t = maximum(frames[1])
    for frame in frames
        min_t = min(min_t, minimum(frame))
        max_t = max(max_t, maximum(frame))
    end

    dir = dirname(path)
    mkpath(dir)

    anim = Plots.Animation()
    for (frame, step) in zip(frames, frame_steps)
        p = Plots.heatmap(frame;
            c=color,
            clims=(min_t, max_t),
            aspect_ratio=1,
            axis=nothing,
            title="Heat propagation (step=$step)",
            colorbar_title="Temperature",
        )
        Plots.frame(anim, p)
    end

    Plots.gif(anim, path; fps=fps)
    return path
end

function run_pipeline(initial::AbstractMatrix{<:Real};
    steps::Int=200,
    alpha::Real=0.2,
    block_h::Int=64,
    block_w::Int=64,
    boundary=:pad,
    pad_value::Real=0.0,
)
    serial = heat_propagate_cpu_serial(initial;
        steps=steps,
        alpha=alpha,
        boundary=boundary,
        pad_value=pad_value,
    )
    dagger = heat_propagate_dagger_stencil(initial;
        steps=steps,
        alpha=alpha,
        block_h=block_h,
        block_w=block_w,
        boundary=boundary,
        pad_value=pad_value,
        return_darray=false,
    )
    err = maximum(abs.(serial .- dagger))
    return serial, dagger, err
end

end
