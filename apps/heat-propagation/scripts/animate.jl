using DaggerHeatPropagation

rows = parse(Int, get(ENV, "HEAT_ROWS", "256"))
cols = parse(Int, get(ENV, "HEAT_COLS", "256"))
steps = parse(Int, get(ENV, "HEAT_STEPS", "400"))
alpha = parse(Float64, get(ENV, "HEAT_ALPHA", "0.2"))
block_h = parse(Int, get(ENV, "HEAT_BLOCK_H", "64"))
block_w = parse(Int, get(ENV, "HEAT_BLOCK_W", "64"))
ambient = parse(Float64, get(ENV, "HEAT_AMBIENT", "0.0"))
hotspot_temperature = parse(Float64, get(ENV, "HEAT_HOTSPOT_TEMP", "1.0"))
hotspot_radius = parse(Int, get(ENV, "HEAT_HOTSPOT_RADIUS", "8"))
boundary = lowercase(strip(get(ENV, "HEAT_BOUNDARY", "pad")))
pad_value = parse(Float64, get(ENV, "HEAT_PAD_VALUE", string(ambient)))
snapshot_every = parse(Int, get(ENV, "HEAT_SNAPSHOT_EVERY", "4"))
fps = parse(Int, get(ENV, "HEAT_FPS", "20"))
out_path = abspath(get(ENV, "HEAT_GIF", joinpath(@__DIR__, "..", "heat_propagation.gif")))

path = DaggerHeatPropagation.save_heat_animation(out_path;
    rows=rows,
    cols=cols,
    steps=steps,
    alpha=alpha,
    block_h=block_h,
    block_w=block_w,
    ambient=ambient,
    hotspot_temperature=hotspot_temperature,
    hotspot_radius=hotspot_radius,
    boundary=boundary,
    pad_value=pad_value,
    snapshot_every=snapshot_every,
    fps=fps,
)

println("Heat animation saved to: $path")
