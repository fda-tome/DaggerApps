#!/bin/bash -l
#PBS -A dagger
#PBS -N barnes_scaling_12h
#PBS -q prod
#PBS -l select=256
#PBS -l walltime=12:00:00
#PBS -l filesystems=flare
#PBS -l place=scatter
#PBS -o /lus/flare/projects/dagger/paper/DaggerApps/benchmarks/results/barnes-hut/pbs/barnes_scaling_12h.out
#PBS -e /lus/flare/projects/dagger/paper/DaggerApps/benchmarks/results/barnes-hut/pbs/barnes_scaling_12h.err


echo "Job ID: ${PBS_JOBID:-unknown}"
echo "Start: $(date)"
echo "Working dir: ${PBS_O_WORKDIR:-$PWD}"

cd "/flare/dagger/paper/DaggerApps"

export BARNES_N_STRONG=2000000
export BARNES_BODIES_PER_PROC=8000
export BENCH_WARMUP=1
export BENCH_RUNS=5
export BENCH_BT_SAMPLES=10
export BENCH_BT_EVALS=1
export BARNES_SCALING_EXPERIMENT=1

julia --project="/flare/dagger/paper/DaggerApps/apps/barnes-hut" -e 'include("/flare/dagger/paper/DaggerApps/benchmarks/scripts/barnes-hut.jl"); run_scaling_experiment(worker_counts=[1,2,4,8,16,32,64,128,256])'

echo "End: $(date)"
