"""
Distributed Barnes-Hut N-body simulation using Dagger.jl and Morton Z-curves.
No boundary conditions; partition once by Morton (no repartition).
"""
module DaggerBarnesHut

using Dagger
using Distributed
using LinearAlgebra
using Random
using StaticArrays

export bmark, run_timesteps, morton_encode, ParticleSnapshot

# -----------------------------------------------------------------------------
# Morton Z-order encoding (3D -> 1D)
# 21 bits per dimension -> 63-bit code; coordinates normalized to [0,1]³
# ------------------------------------------------------------------------------

const MORTON_BITS_PER_DIM = 21
const MORTON_MAX = (1 << (3 * MORTON_BITS_PER_DIM)) - 1

"""Spread 21 bits of x into positions 0,3,6,... (bit interleave for Morton)."""
@inline function _spread3(x::UInt32)::UInt64
    u = UInt64(x) & 0x1fffff  # 21 bits
    u = (u | (u << 32)) & 0x1f00000000ffff  # ...
    u = (u | (u << 16)) & 0x1f0000ff0000ff
    u = (u | (u << 8))  & 0x100f00f00f00f00f
    u = (u | (u << 4))  & 0x10c30c30c30c30c3
    u = (u | (u << 2))  & 0x1249249249249249
    return u
end

"""
    morton_encode(x, y, z) -> UInt64

Encode 3D coordinates (in [0,1]) to a 63-bit Morton code.
Clamps to [0,1] and quantizes to 21 bits per dimension.
"""
function morton_encode(x::Real, y::Real, z::Real)::UInt64
    # Clamp to [0,1] and scale to 21-bit integer; cap at 2^21-1 to avoid overflow
    scale = Float64(1 << MORTON_BITS_PER_DIM)
    raw_x = trunc(Int, clamp(Float64(x), 0.0, 1.0) * scale)
    raw_y = trunc(Int, clamp(Float64(y), 0.0, 1.0) * scale)
    raw_z = trunc(Int, clamp(Float64(z), 0.0, 1.0) * scale)
    ux = UInt32(min(raw_x, 0x1fffff))
    uy = UInt32(min(raw_y, 0x1fffff))
    uz = UInt32(min(raw_z, 0x1fffff))
    return _spread3(ux) | (_spread3(uy) << 1) | (_spread3(uz) << 2)
end

# -----------------------------------------------------------------------------
# Particle data (SoA)
# ------------------------------------------------------------------------------

struct ParticleSnapshot
    pos::Matrix{Float64}   # 3 x N
    vel::Matrix{Float64}   # 3 x N
    mass::Vector{Float64}  # N
end

function ParticleSnapshot(N::Int)
    pos = zeros(3, N)
    vel = zeros(3, N)
    mass = ones(N)
    return ParticleSnapshot(pos, vel, mass)
end

Base.size(ps::ParticleSnapshot) = size(ps.pos, 2)

"""Initialize particles in unit cube with random positions and zero velocity."""
function random_particles(N::Int; seed::Union{Int,Nothing}=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    pos = rand(rng, 3, N)
    vel = zeros(3, N)
    mass = ones(N)
    return ParticleSnapshot(pos, vel, mass)
end

# -----------------------------------------------------------------------------
# Octree (local to a worker; monopole only for simplicity)
# ------------------------------------------------------------------------------

mutable struct OctNode
    geom_center::SVector{3,Float64}  # geometric center of the cell
    center::SVector{3,Float64}
    mass::Float64
    size::Float64  # side length of cell
    children::Union{Nothing,Vector{OctNode}}  # length 8 or nothing for leaf
    leaf_indices::Vector{Int}  # particle indices in this leaf (empty if internal)
end

function OctNode(geom_center::SVector{3,Float64}, center::SVector{3,Float64}, mass::Float64, size::Float64)
    OctNode(geom_center, center, mass, size, nothing, Int[])
end

function OctNode(geom_center::SVector{3,Float64}, center::SVector{3,Float64}, mass::Float64, size::Float64, indices::Vector{Int})
    OctNode(geom_center, center, mass, size, nothing, indices)
end

function is_leaf(n::OctNode)
    n.children === nothing
end

@inline function _node_contains_index(node::OctNode, idx::Int)::Bool
    if is_leaf(node)
        return idx in node.leaf_indices
    end
    @inbounds for ch in node.children
        _node_contains_index(ch, idx) && return true
    end
    return false
end

# Build octree from Morton-sorted particle indices (positions and masses)
function build_octree!(pos::AbstractMatrix{Float64}, mass::Vector{Float64}, sorted_indices::AbstractVector{Int}, lo::Int, hi::Int, box_lo::SVector{3,Float64}, box_size::Float64)::OctNode
    n = hi - lo + 1
    if n <= 0
        gc = box_lo + SVector(box_size / 2, box_size / 2, box_size / 2)
        return OctNode(gc, gc, 0.0, box_size)
    end
    if n == 1
        i = sorted_indices[lo]
        c = SVector(pos[1,i], pos[2,i], pos[3,i])
        return OctNode(c, c, mass[i], box_size, [i])
    end
    # Compute center of mass and total mass
    m_tot = 0.0
    cx = cy = cz = 0.0
    for k in lo:hi
        i = sorted_indices[k]
        m_tot += mass[i]
        cx += pos[1,i] * mass[i]
        cy += pos[2,i] * mass[i]
        cz += pos[3,i] * mass[i]
    end
    inv_m = 1.0 / m_tot
    center = SVector(cx * inv_m, cy * inv_m, cz * inv_m)
    node = OctNode(box_lo + SVector(box_size / 2, box_size / 2, box_size / 2), center, m_tot, box_size)
    # Subdivide into 8 octants (Morton order: 0..7)
    half = box_size / 2
    children = Vector{OctNode}(undef, 8)
    for oct in 0:7
        # Octant bounds in Morton order: x then y then z
        x0 = (oct & 1) != 0
        y0 = (oct & 2) != 0
        z0 = (oct & 4) != 0
        o_lo = SVector(box_lo[1] + (x0 ? half : 0.0), box_lo[2] + (y0 ? half : 0.0), box_lo[3] + (z0 ? half : 0.0))
        # Count and partition indices into this octant
        start = lo
        for k in lo:hi
            i = sorted_indices[k]
            px, py, pz = pos[1,i], pos[2,i], pos[3,i]
            in_x = (px >= o_lo[1] && px < o_lo[1] + half) || (half == 0 && px <= o_lo[1])
            in_y = (py >= o_lo[2] && py < o_lo[2] + half) || (half == 0 && py <= o_lo[2])
            in_z = (pz >= o_lo[3] && pz < o_lo[3] + half) || (half == 0 && pz <= o_lo[3])
            if in_x && in_y && in_z
                sorted_indices[k], sorted_indices[start] = sorted_indices[start], sorted_indices[k]
                start += 1
            end
        end
        count_oct = start - lo
        if count_oct > 0
            children[oct+1] = build_octree!(pos, mass, sorted_indices, lo, start - 1, o_lo, half)
        else
            gc = o_lo + SVector(half / 2, half / 2, half / 2)
            children[oct+1] = OctNode(gc, gc, 0.0, half)
        end
        lo = start
    end
    node.children = children
    return node
end

# Simpler: build from sorted index range by recursive split along longest axis (KD-style) to avoid complex octant partitioning
function _split_sorted!(pos::AbstractMatrix{Float64}, mass::Vector{Float64}, idx::Vector{Int}, lo::Int, hi::Int, box_lo::SVector{3,Float64}, box_size::Float64)::OctNode
    geom_center = box_lo + SVector(box_size / 2, box_size / 2, box_size / 2)
    n = hi - lo + 1
    if n <= 0
        return OctNode(geom_center, geom_center, 0.0, box_size)
    end
    LEAF_CAPACITY = 16
    m_tot = 0.0
    cx = cy = cz = 0.0
    for k in lo:hi
        i = idx[k]
        m_tot += mass[i]
        cx += pos[1,i] * mass[i]
        cy += pos[2,i] * mass[i]
        cz += pos[3,i] * mass[i]
    end
    inv_m = 1.0 / m_tot
    com = SVector(cx * inv_m, cy * inv_m, cz * inv_m)
    if n <= LEAF_CAPACITY
        return OctNode(geom_center, com, m_tot, box_size, idx[lo:hi])
    end
    node = OctNode(geom_center, com, m_tot, box_size)
    # Split along axis with largest data extent at median.
    xmin = ymin = zmin = Inf
    xmax = ymax = zmax = -Inf
    @inbounds for k in lo:hi
        i = idx[k]
        x = pos[1, i]
        y = pos[2, i]
        z = pos[3, i]
        xmin = min(xmin, x); xmax = max(xmax, x)
        ymin = min(ymin, y); ymax = max(ymax, y)
        zmin = min(zmin, z); zmax = max(zmax, z)
    end
    ext = (xmax - xmin, ymax - ymin, zmax - zmin)
    ax = argmax(ext)
    mid = (lo + hi) >> 1
    # Partial sort to get median along axis ax
    segment = copy(@view idx[lo:hi])
    ax_vals = [pos[ax, segment[p]] for p in 1:length(segment)]
    perm = sortperm(ax_vals)
    @inbounds for (p, k) in enumerate(lo:hi)
        idx[k] = segment[perm[p]]
    end
    half = box_size / 2
    o_lo_low = box_lo
    o_lo_high = SVector(
        ax == 1 ? box_lo[1] + half : box_lo[1],
        ax == 2 ? box_lo[2] + half : box_lo[2],
        ax == 3 ? box_lo[3] + half : box_lo[3]
    )
    left = _split_sorted!(pos, mass, idx, lo, mid, o_lo_low, half)
    right = _split_sorted!(pos, mass, idx, mid+1, hi, o_lo_high, half)
    node.children = [left, right]
    return node
end

# Build octree from particles (positions, mass) and Morton-sorted indices
function build_octree(pos::AbstractMatrix{Float64}, mass::Vector{Float64}; sorted_indices::Union{AbstractVector{Int},Nothing}=nothing)
    N = size(pos, 2)
    idx = sorted_indices === nothing ? collect(1:N) : copy(collect(sorted_indices))
    if length(idx) == 0
        gc = SVector(0.5, 0.5, 0.5)
        return OctNode(gc, gc, 0.0, 1.0)
    end
    box_lo = SVector(0.0, 0.0, 0.0)
    box_size = 1.0
    return _split_sorted!(pos, mass, idx, 1, length(idx), box_lo, box_size)
end

# -----------------------------------------------------------------------------
# Force with theta criterion (monopole, softening)
# ------------------------------------------------------------------------------

const DEFAULT_SOFTENING = 1e-10

function force_from_node!(acc::Vector{Float64}, pos_i::SVector{3,Float64}, node::OctNode, theta::Float64, G::Float64, eps2::Float64, pos::AbstractMatrix{Float64}, mass::Vector{Float64}, skip_index::Int)
    r = node.center - pos_i
    d2 = max(dot(r, r), eps2)
    d = sqrt(d2)
    if is_leaf(node)
        for idx in node.leaf_indices
            idx == skip_index && continue
            r_i = SVector(pos[1,idx], pos[2,idx], pos[3,idx]) - pos_i
            d2_i = max(dot(r_i, r_i), eps2)
            d_i = sqrt(d2_i)
            fac = G * mass[idx] / d2_i
            inv_d = 1.0 / d_i
            acc[1] += fac * r_i[1] * inv_d
            acc[2] += fac * r_i[2] * inv_d
            acc[3] += fac * r_i[3] * inv_d
        end
        return
    end
    # Internal node: stricter COM-offset criterion.
    # Never accept a node that still contains the target particle.
    # Use d_eff = d - ||COM - geometric_center|| to penalize lopsided nodes.
    delta = norm(node.center - node.geom_center)
    d_eff = max(d - delta, sqrt(eps2))
    if !_node_contains_index(node, skip_index) && node.size / d_eff <= theta
        f = G * node.mass / d2
        inv_d = 1.0 / d
        acc[1] += f * r[1] * inv_d
        acc[2] += f * r[2] * inv_d
        acc[3] += f * r[3] * inv_d
        return
    end
    for ch in node.children
        if ch.mass > 0
            force_from_node!(acc, pos_i, ch, theta, G, eps2, pos, mass, skip_index)
        end
    end
end

# Multipole record for global directory (Phase 3)
struct MultipoleRecord
    center::SVector{3,Float64}
    mass::Float64
    size::Float64
end

function force_from_multipole!(acc::Vector{Float64}, pos_i::SVector{3,Float64}, mp::MultipoleRecord, G::Float64, eps2::Float64)
    r = mp.center - pos_i
    d2 = max(dot(r, r), eps2)
    d = sqrt(d2)
    f = G * mp.mass / d2
    inv_d = 1.0 / d
    acc[1] += f * r[1] * inv_d
    acc[2] += f * r[2] * inv_d
    acc[3] += f * r[3] * inv_d
end

function compute_acceleration(pos::AbstractMatrix{Float64}, mass::Vector{Float64}, tree::OctNode, i::Int, theta::Float64; G=1.0, eps2=DEFAULT_SOFTENING^2)
    acc = zeros(3)
    pos_i = SVector(pos[1,i], pos[2,i], pos[3,i])
    force_from_node!(acc, pos_i, tree, theta, G, eps2, pos, mass, i)
    return SVector(acc[1], acc[2], acc[3])
end

# -----------------------------------------------------------------------------
# Integration (leapfrog or Euler)
# ------------------------------------------------------------------------------

function step!(ps::ParticleSnapshot, tree::OctNode, dt::Float64, theta::Float64; G=1.0, eps2=DEFAULT_SOFTENING^2)
    pos, vel, mass = ps.pos, ps.vel, ps.mass
    N = size(pos, 2)
    for i in 1:N
        acc = compute_acceleration(pos, mass, tree, i, theta; G=G, eps2=eps2)
        vel[1,i] += acc[1] * dt
        vel[2,i] += acc[2] * dt
        vel[3,i] += acc[3] * dt
        pos[1,i] += vel[1,i] * dt
        pos[2,i] += vel[2,i] * dt
        pos[3,i] += vel[3,i] * dt
    end
    return ps
end

# -----------------------------------------------------------------------------
# Serial path: one process, one tree (Phase 1)
# ------------------------------------------------------------------------------

function run_timesteps_serial(ps::ParticleSnapshot, nsteps::Int, dt::Float64, theta::Float64; G=1.0)
    pos = ps.pos
    mass = ps.mass
    N = size(pos, 2)
    # Morton sort
    codes = [morton_encode(pos[1,i], pos[2,i], pos[3,i]) for i in 1:N]
    perm = sortperm(codes)
    sorted_indices = perm
    tree = build_octree(pos, mass; sorted_indices=sorted_indices)
    for _ in 1:nsteps
        step!(ps, tree, dt, theta; G=G)
        tree = build_octree(pos, mass; sorted_indices=sorted_indices)
    end
    return ps
end

# -----------------------------------------------------------------------------
# Distributed path: partition by Morton, per-worker octree, global multipole directory (Phase 2 + 3)
# ------------------------------------------------------------------------------

function _worker_segment_range(N::Int, nworkers::Int, w::Int)
    base = div(N, nworkers)
    rem = N % nworkers
    start = 1 + (w - 1) * base + min(w - 1, rem)
    stop = start + base + (w <= rem ? 1 : 0) - 1
    return start:stop
end

"""
    bmark(N, theta; nsteps=1, dt=0.01)

Run N-body Barnes-Hut benchmark: N particles, theta opening angle.
Returns nothing (timed by benchmark script). Uses distributed path when workers exist.
"""
function bmark(N::Int, theta::Float64; nsteps::Int=1, dt::Float64=0.01)
    nworkers = nprocs()
    if nworkers <= 1
        ps = random_particles(N; seed=42)
        run_timesteps_serial(ps, nsteps, dt, theta)
        return nothing
    end
    # Distributed: run on worker 1 as driver for simplicity; workers 2..n run tasks
    # Dagger expects workers to be in the scheduler; we use the current process + workers()
    workers_list = workers()
    if isempty(workers_list)
        ps = random_particles(N; seed=42)
        run_timesteps_serial(ps, nsteps, dt, theta)
        return nothing
    end
    nw = length(workers_list)
    # Create particle data on driver
    ps_global = random_particles(N; seed=42)
    pos = ps_global.pos
    vel = ps_global.vel
    mass = ps_global.mass
    # Morton sort on driver
    codes = [morton_encode(pos[1,i], pos[2,i], pos[3,i]) for i in 1:N]
    perm = sortperm(codes)
    sorted_indices = perm
    # Partition: each worker gets contiguous segment of sorted indices
    segments = Vector{UnitRange{Int}}(undef, nw)
    for (iw, w) in enumerate(workers_list)
        seg = _worker_segment_range(N, nw, iw)
        segments[iw] = seg
    end
    # Build per-worker segments on driver (only that slice is sent to each worker)
    segs = Vector{Any}(undef, nw)
    for iw in 1:nw
        seg = _worker_segment_range(N, nw, iw)
        nseg = length(seg)
        pos_w = zeros(3, nseg)
        vel_w = zeros(3, nseg)
        mass_w = zeros(nseg)
        for (j, i) in enumerate(seg)
            idx = sorted_indices[i]
            for d in 1:3
                pos_w[d, j] = pos[d, idx]
                vel_w[d, j] = vel[d, idx]
            end
            mass_w[j] = mass[idx]
        end
        segs[iw] = (pos=pos_w, vel=vel_w, mass=mass_w)
    end
    # Per-worker: build local octree from segment, then we need global multipole directory
    # Gather multipoles from each worker (root of local tree = one multipole per worker for coarsest level)
    function build_local_octree(seg)
        pos_w = seg.pos
        mass_w = seg.mass
        tree = build_octree(pos_w, mass_w)
        root_mp = MultipoleRecord(tree.center, tree.mass, tree.size)
        (tree=tree, root_mp=root_mp)
    end
    local_tasks = [Dagger.@spawn scope=Dagger.scope(worker=w) build_local_octree(segs[iw]) for (iw, w) in enumerate(workers_list)]
    local_results = fetch.(local_tasks)
    # Global multipole directory: list of root multipoles from each worker
    global_directory = [r.root_mp for r in local_results]
    # One timestep: each worker computes forces for its particles using local tree + global_directory for "remote" (treat directory as single level of multipoles)
    function force_and_step(seg, tree_root_mp, all_root_mps, my_worker_rank, dt_val, theta_val)
        pos_w = seg.pos
        vel_w = seg.vel
        mass_w = seg.mass
        tree = tree_root_mp.tree
        nseg = size(pos_w, 2)
        G = 1.0
        eps2 = DEFAULT_SOFTENING^2
        for j in 1:nseg
            acc = zeros(3)
            pos_i = SVector(pos_w[1,j], pos_w[2,j], pos_w[3,j])
            # Local tree (includes self-avoidance via skip_index)
            force_from_node!(acc, pos_i, tree, theta_val, G, eps2, pos_w, mass_w, j)
            # Remote: other workers' root multipoles only (exclude own to avoid double count)
            for (iw, mp) in enumerate(all_root_mps)
                iw == my_worker_rank && continue
                if mp.mass > 0
                    force_from_multipole!(acc, pos_i, mp, G, eps2)
                end
            end
            vel_w[1,j] += acc[1] * dt_val
            vel_w[2,j] += acc[2] * dt_val
            vel_w[3,j] += acc[3] * dt_val
            pos_w[1,j] += vel_w[1,j] * dt_val
            pos_w[2,j] += vel_w[2,j] * dt_val
            pos_w[3,j] += vel_w[3,j] * dt_val
        end
        (pos=pos_w, vel=vel_w, mass=mass_w)
    end
    # Broadcast global_directory to all workers (it's small)
    for _ in 1:(nsteps - 1)
        # Rebuild local trees after position update
        local_tasks = [Dagger.@spawn scope=Dagger.scope(worker=w) build_local_octree(segs[iw]) for (iw, w) in enumerate(workers_list)]
        local_results = fetch.(local_tasks)
        global_directory = [r.root_mp for r in local_results]
        step_tasks = [Dagger.@spawn scope=Dagger.scope(worker=w) force_and_step(segs[iw], local_results[iw], global_directory, iw, dt, theta) for (iw, w) in enumerate(workers_list)]
        segs = fetch.(step_tasks)
    end
    # Last step: same
    local_tasks = [Dagger.@spawn scope=Dagger.scope(worker=w) build_local_octree(segs[iw]) for (iw, w) in enumerate(workers_list)]
    local_results = fetch.(local_tasks)
    global_directory = [r.root_mp for r in local_results]
    step_tasks = [Dagger.@spawn scope=Dagger.scope(worker=w) force_and_step(segs[iw], local_results[iw], global_directory, iw, dt, theta) for (iw, w) in enumerate(workers_list)]
    fetch.(step_tasks)
    return nothing
end

"""Run a fixed number of timesteps (serial or distributed via bmark)."""
function run_timesteps(ps::ParticleSnapshot, nsteps::Int, dt::Float64, theta::Float64; G=1.0)
    if nprocs() <= 1 || isempty(workers())
        return run_timesteps_serial(ps, nsteps, dt, theta; G=G)
    end
    # For run_timesteps with existing ps we'd need to distribute it; bmark creates its own
    run_timesteps_serial(ps, nsteps, dt, theta; G=G)
end

end # module
