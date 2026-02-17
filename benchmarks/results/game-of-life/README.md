# Game-of-Life benchmark results

Benchmark outputs for game-of-life are written under this folder, typically in timestamped subfolders like:

```
benchmarks/results/game-of-life/<timestamp>/
```

Each run writes:

- `strong_scaling.csv`
- `weak_scaling.csv`
- optional GPU size-sweep outputs under `gpu_size_sweep/`:
  - `gpu_size_sweep_runs.csv`
  - `gpu_size_sweep_summary.csv`
  - `gpu_size_sweep.png`

Columns:

```
scenario,variant,device,threads,rows,cols,steps,block_h,block_w,density,run,time_sec
```
