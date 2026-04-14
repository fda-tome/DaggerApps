#!/bin/bash -l
#PBS -A dagger
#PBS -N barnes_test
#PBS -q prod
#PBS -l select=256
#PBS -l walltime=00:30:00
#PBS -l filesystems=flare
#PBS -l place=scatter
#PBS -o /lus/flare/projects/dagger/paper/DaggerApps/benchmarks/results/barnes-hut/pbs/barnes_test.out
#PBS -e /lus/flare/projects/dagger/paper/DaggerApps/benchmarks/results/barnes-hut/pbs/barnes_test.err

echo "Job ID: ${PBS_JOBID:-unknown}"
echo "Start: $(date)"
echo "Working dir: ${PBS_O_WORKDIR:-$PWD}"
echo "PBS_NODEFILE: ${PBS_NODEFILE:-unset}"
echo "Unique nodes: $(sort -u "$PBS_NODEFILE" 2>/dev/null | wc -l)"

cd "/flare/dagger/paper/DaggerApps"

export BARNES_N_STRONG=10000
export BARNES_BODIES_PER_PROC=1000
export BENCH_WARMUP=1
export BENCH_RUNS=1
export BENCH_BT_SAMPLES=1
export BENCH_BT_EVALS=1
export BARNES_SCALING_EXPERIMENT=1

julia --project="/flare/dagger/paper/DaggerApps/apps/barnes-hut" -e 'include("/flare/dagger/paper/DaggerApps/benchmarks/scripts/barnes-hut.jl"); run_scaling_experiment(worker_counts=[1,2,4,8,16,32,64,128,256])'

echo "End: $(date)"
