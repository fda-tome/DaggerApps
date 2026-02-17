# Results

Benchmarks write outputs here **per app**:

- Results for app `<app>` go under `benchmarks/results/<app>/` (where `<app>` matches the folder name under `apps/`).
- Each run typically writes a timestamped subfolder.
- Keep large binary artifacts out of version control when possible; link to external storage if needed.

## CSV formats

Seam‑carving (`benchmarks/results/seam-carving/`):

- `strong_scaling.csv` / `weak_scaling.csv`
- Columns: `scenario,variant,threads,rows,cols,k,tile_h,tile_w,run,time_sec`

Barnes–Hut (`benchmarks/results/barnes-hut/`):

- `strong_scaling.csv` / `weak_scaling.csv`
- Columns: `scenario,dagger_processors,N,theta,run,time_sec`

Game-of-life (`benchmarks/results/game-of-life/`):

- `strong_scaling.csv` / `weak_scaling.csv`
- Columns: `scenario,variant,device,threads,rows,cols,steps,block_h,block_w,density,run,time_sec`
- GPU size sweep (optional): `gpu_size_sweep/gpu_size_sweep_runs.csv`, `gpu_size_sweep/gpu_size_sweep_summary.csv`, `gpu_size_sweep/gpu_size_sweep.png`

Heat-propagation (`benchmarks/results/heat-propagation/`):

- `strong_scaling.csv` / `weak_scaling.csv`
- Columns: `scenario,variant,device,threads,rows,cols,steps,block_h,block_w,alpha,ambient,hotspot_temp,hotspot_radius,pad_value,run,time_sec`
- GPU size sweep (optional): `gpu_size_sweep/gpu_size_sweep_runs.csv`, `gpu_size_sweep/gpu_size_sweep_summary.csv`, `gpu_size_sweep/gpu_size_sweep.png`
