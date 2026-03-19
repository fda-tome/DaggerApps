using Random
using LinearAlgebra
using Statistics

# Load your implementation
include("/home/felipetome/dagger-pass@k/DaggerApps/apps/barnes-hut/src/DaggerBarnesHut.jl")
using Main.DaggerBarnesHut

const EPS = 1e-10
const G = 1.0

# ---------------------------
# Direct-sum accelerations
# ---------------------------
function direct_accelerations(pos::Matrix{Float64}, mass::Vector{Float64}; eps2=EPS^2, G=G)
    N = size(pos, 2)
    acc = zeros(3, N)
    @inbounds for i in 1:N
        xi = @view pos[:, i]
        aix = 0.0; aiy = 0.0; aiz = 0.0
        for j in 1:N
            i == j && continue
            dx = pos[1, j] - xi[1]
            dy = pos[2, j] - xi[2]
            dz = pos[3, j] - xi[3]
            r2 = max(dx*dx + dy*dy + dz*dz, eps2)
            invr = inv(sqrt(r2))
            fac = G * mass[j] * invr / r2
            aix += fac * dx
            aiy += fac * dy
            aiz += fac * dz
        end
        acc[1, i] = aix
        acc[2, i] = aiy
        acc[3, i] = aiz
    end
    return acc
end

# ---------------------------
# BH accelerations via your tree
# ---------------------------
function bh_accelerations(pos::Matrix{Float64}, mass::Vector{Float64}, theta::Float64)
    N = size(pos, 2)
    codes = [DaggerBarnesHut.morton_encode(pos[1,i], pos[2,i], pos[3,i]) for i in 1:N]
    perm = sortperm(codes)
    tree = DaggerBarnesHut.build_octree(pos, mass; sorted_indices=perm)
    acc = zeros(3, N)
    @inbounds for i in 1:N
        ai = DaggerBarnesHut.compute_acceleration(pos, mass, tree, i, theta; G=G, eps2=EPS^2)
        acc[1, i] = ai[1]
        acc[2, i] = ai[2]
        acc[3, i] = ai[3]
    end
    return acc
end

# ---------------------------
# Error metrics
# ---------------------------
function rel_errors(acc_bh::Matrix{Float64}, acc_ref::Matrix{Float64})
    N = size(acc_ref, 2)
    errs = zeros(N)
    @inbounds for i in 1:N
        num = norm(acc_bh[:, i] .- acc_ref[:, i])
        den = norm(acc_ref[:, i]) + 1e-12
        errs[i] = num / den
    end
    return errs
end

function summarize(errs::Vector{Float64})
    return (
        mean = mean(errs),
        median = median(errs),
        p95 = quantile(errs, 0.95),
        max = maximum(errs),
    )
end

# ---------------------------
# Run force-accuracy study
# ---------------------------
function run_force_accuracy(; Ns=[128, 256, 512], thetas=[0.8, 0.5, 0.3, 0.2], seed=42)
    println("=== Force accuracy vs direct-sum ===")
    println("N,theta,mean,median,p95,max")
    for N in Ns
        rng = MersenneTwister(seed + N)
        pos = rand(rng, 3, N)
        mass = ones(Float64, N)

        acc_ref = direct_accelerations(pos, mass)
        for θ in thetas
            acc_bh = bh_accelerations(pos, mass, θ)
            errs = rel_errors(acc_bh, acc_ref)
            s = summarize(errs)
            println("$(N),$(θ),$(s.mean),$(s.median),$(s.p95),$(s.max)")
        end
    end
end

# ---------------------------
# Optional: short integration sanity
# ---------------------------
function kinetic_energy(pos, vel, mass)
    N = length(mass)
    k = 0.0
    @inbounds for i in 1:N
        v2 = vel[1,i]^2 + vel[2,i]^2 + vel[3,i]^2
        k += 0.5 * mass[i] * v2
    end
    k
end

function potential_energy(pos, mass; eps2=EPS^2, G=G)
    N = length(mass)
    u = 0.0
    @inbounds for i in 1:N-1
        for j in i+1:N
            dx = pos[1,j] - pos[1,i]
            dy = pos[2,j] - pos[2,i]
            dz = pos[3,j] - pos[3,i]
            r = sqrt(max(dx*dx + dy*dy + dz*dz, eps2))
            u += -G * mass[i] * mass[j] / r
        end
    end
    u
end

function total_momentum(vel, mass)
    p = zeros(3)
    N = length(mass)
    @inbounds for i in 1:N
        p[1] += mass[i] * vel[1,i]
        p[2] += mass[i] * vel[2,i]
        p[3] += mass[i] * vel[3,i]
    end
    p
end

function run_short_integration_sanity(; N=1000, steps=100, theta=0.5, dt=1e-3, seed=7)
    println("\n=== Short integration sanity ===")
    rng = MersenneTwister(seed)
    pos = rand(rng, 3, N)
    vel = zeros(3, N)
    mass = ones(Float64, N)

    ps = DaggerBarnesHut.ParticleSnapshot(copy(pos), copy(vel), copy(mass))

    E0 = kinetic_energy(ps.pos, ps.vel, ps.mass) + potential_energy(ps.pos, ps.mass)
    P0 = total_momentum(ps.vel, ps.mass)

    # simple repeated single-step path
    for _ in 1:steps
        codes = [DaggerBarnesHut.morton_encode(ps.pos[1,i], ps.pos[2,i], ps.pos[3,i]) for i in 1:N]
        perm = sortperm(codes)
        tree = DaggerBarnesHut.build_octree(ps.pos, ps.mass; sorted_indices=perm)
        DaggerBarnesHut.step!(ps, tree, dt, theta)
    end

    E1 = kinetic_energy(ps.pos, ps.vel, ps.mass) + potential_energy(ps.pos, ps.mass)
    P1 = total_momentum(ps.vel, ps.mass)

    dE = abs(E1 - E0) / (abs(E0) + 1e-12)
    dP = norm(P1 - P0) / (norm(P0) + 1e-12)

    println("N=$N, steps=$steps, theta=$theta, dt=$dt")
    println("relative energy drift: $dE")
    println("relative momentum drift: $dP")
end

# Main
run_force_accuracy()
run_short_integration_sanity()
