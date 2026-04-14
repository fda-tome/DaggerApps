# Westrick: fixed SEAM_WESTRICK_BLOCK_WIDTH_SCHED_THREADS ⇒ same block_width for every julia -t.
#
# From paper repo root (expect identical "bw=" for both):
#   julia --project=/flare/dagger/paper/DaggerApps/apps/seam-carving -t1  /flare/dagger/paper/DaggerApps/apps/seam-carving/test/westrick_bw_sched_threads.jl
#   julia --project=/flare/dagger/paper/DaggerApps/apps/seam-carving -t52 /flare/dagger/paper/DaggerApps/apps/seam-carving/test/westrick_bw_sched_threads.jl

using DaggerSeamCarving

const W = 10400

withenv(
    "SEAM_WESTRICK_BLOCK_WIDTH_AUTO" => "1",
    "SEAM_WESTRICK_BLOCK_WIDTH_SCHED_THREADS" => "52",
    "SEAM_WESTRICK_BLOCK_WIDTH_AUTO_TASK_TARGET" => "28",
) do
    bw = DaggerSeamCarving._resolve_westrick_block_width(W, nothing)
    numB = 1 + (W - 1) ÷ bw
    Tsched, _ = DaggerSeamCarving._westrick_auto_sched_T(52)
    numB >= Tsched || error("numB=$numB < Tsched=$Tsched (infeasible width)")
    println("OK bw=", bw, " numB=", numB, " Tsched=", Tsched, " nthreads=", Threads.nthreads())
end
