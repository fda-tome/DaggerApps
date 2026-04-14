# End-to-end correctness check: cpu_dagger_westrick vs cpu_serial.
#
# Verifies that the Dagger-parallel Westrick pipeline produces bitwise-identical
# results to the serial reference at every stage: energy, DP (M,B), seam path,
# and final carved image.
#
# Run with multiple threads to actually exercise the parallel paths:
#
#   julia --project=/flare/dagger/paper/DaggerApps/apps/seam-carving -t8 \
#     /flare/dagger/paper/DaggerApps/apps/seam-carving/test/westrick_correctness.jl
#
# Or with the exact benchmark settings:
#
#   SEAM_WESTRICK_BLOCK_WIDTH_AUTO=1 \
#   SEAM_WESTRICK_BLOCK_WIDTH_SCHED_THREADS=52 \
#   SEAM_WESTRICK_BLOCK_WIDTH_AUTO_TASK_TARGET=6 \
#   SEAM_WESTRICK_BLOCK_WIDTH_AUTO_MAX=4096 \
#   julia --project=/flare/dagger/paper/DaggerApps/apps/seam-carving -t8 \
#     /flare/dagger/paper/DaggerApps/apps/seam-carving/test/westrick_correctness.jl

using DaggerSeamCarving
using Random

const DSC = DaggerSeamCarving

function test_correctness(; H::Int, W::Int, k::Int, tile_h::Int, tile_w::Int,
                            block_width::Union{Nothing,Int}, seed::Int)
    rng = MersenneTwister(seed)
    img = rand(rng, Float32, H, W)

    println("--- Correctness test: $(H)×$(W), k=$k, tile=$(tile_h)×$(tile_w), ",
            "block_width=$(something(block_width, "auto")), ",
            "nthreads=$(Threads.nthreads()), seed=$seed ---")

    # -- Stage 1: Energy --
    E_serial = DSC.energy_cpu_serial(img)
    E_dagger = DSC.energy_cpu_dagger_tiled_serial(img; tile_h, tile_w)
    if E_serial == E_dagger
        println("  [PASS] Energy: bitwise match")
    else
        n_diff = count(E_serial .!= E_dagger)
        max_diff = maximum(abs.(E_serial .- E_dagger))
        println("  [FAIL] Energy: $n_diff / $(length(E_serial)) elements differ, max |diff| = $max_diff")
        return false
    end

    # -- Stage 2: DP (cumulative energy M + backtrack B) --
    M_serial, B_serial = DSC.cumulative_energy_cpu_serial(E_serial)

    bw_resolved = DSC._resolve_westrick_block_width(W, block_width)
    println("  block_width resolved to: $bw_resolved")

    M_westrick, B_westrick = DSC.cumulative_energy_cpu_westrick(E_serial; block_width=bw_resolved)
    if M_serial == M_westrick && B_serial == B_westrick
        println("  [PASS] Westrick Dagger DP: M and B bitwise match serial")
    else
        n_diff_M = count(M_serial .!= M_westrick)
        n_diff_B = count(B_serial .!= B_westrick)
        println("  [FAIL] Westrick Dagger DP: M differs in $n_diff_M, B differs in $n_diff_B elements")
        if n_diff_M > 0
            max_M = maximum(abs.(M_serial .- M_westrick))
            first_row = findfirst(any(M_serial .!= M_westrick; dims=2))[1]
            println("         max |M diff| = $max_M, first differing row = $first_row")
        end
        return false
    end

    # Also check serial Westrick DP (strip schedule without Dagger)
    M_ws, B_ws = DSC.cumulative_energy_cpu_westrick_serial(E_serial; block_width=bw_resolved)
    if M_serial == M_ws && B_serial == B_ws
        println("  [PASS] Westrick serial DP: bitwise match")
    else
        println("  [FAIL] Westrick serial DP: differs from reference")
        return false
    end

    # -- Stage 3: Seam path --
    seam_serial = DSC.find_seam(M_serial, B_serial)
    seam_westrick = DSC.find_seam(M_westrick, B_westrick)
    if seam_serial == seam_westrick
        println("  [PASS] Seam path: identical ($(length(seam_serial)) rows)")
    else
        n_diff = count(seam_serial .!= seam_westrick)
        println("  [FAIL] Seam path: $n_diff / $(length(seam_serial)) rows differ")
        return false
    end

    # -- Stage 4: Seam removal --
    out_serial = DSC.remove_seam_serial(img, seam_serial)
    out_dagger = DSC.remove_seam_dagger_tiled_serial(img, seam_serial; tile_h)
    if out_serial == out_dagger
        println("  [PASS] Seam removal: bitwise match ($(size(out_serial)))")
    else
        n_diff = count(out_serial .!= out_dagger)
        println("  [FAIL] Seam removal: $n_diff / $(length(out_serial)) elements differ")
        return false
    end

    # -- Stage 5: Full pipeline (k seams) --
    img_serial = DSC.seam_carve_cpu_serial(img; k)
    img_westrick = DSC.seam_carve_cpu_dagger_westrick(img; k, tile_h, tile_w, block_width)
    if img_serial == img_westrick
        println("  [PASS] Full pipeline ($k seams): bitwise match ($(size(img_serial)))")
    else
        n_diff = count(img_serial .!= img_westrick)
        max_diff = maximum(abs.(img_serial .- img_westrick))
        println("  [FAIL] Full pipeline ($k seams): $n_diff / $(length(img_serial)) differ, max |diff| = $max_diff")
        return false
    end

    println("  ALL STAGES PASSED")
    return true
end

all_pass = true

for (H, W, k, desc) in [
    (64, 128, 1, "tiny"),
    (200, 300, 3, "small 3-seam"),
    (600, 800, 1, "medium"),
    (600, 800, 5, "medium 5-seam"),
    (2400, 2600, 1, "large"),
]
    ok = test_correctness(; H, W, k, tile_h=180, tile_w=400, block_width=nothing, seed=42)
    if !ok
        global all_pass = false
        println("  STOPPING on first failure.")
        break
    end
    println()
end

if all_pass
    println("\n=== ALL CORRECTNESS TESTS PASSED ===")
else
    println("\n=== CORRECTNESS FAILURE DETECTED ===")
    exit(1)
end
