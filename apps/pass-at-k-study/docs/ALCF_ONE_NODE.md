# Running the study on one node (Aurora or Polaris, ALCF)

Both systems use **PBS Pro**. This doc covers running the pipeline on a single node of **Aurora** (Intel Max GPUs) or **Polaris** (NVIDIA A100 GPUs).

| System  | GPUs per node        | PBS select           | Filesystems   |
|---------|----------------------|----------------------|---------------|
| **Aurora**  | 12 (Intel Max / Ponte Vecchio) | `select=1`           | `home:flare`  |
| **Polaris** | 4 (NVIDIA A100, 40 GB each)    | `select=1:system=polaris` | `home:eagle` |

---

## Option A: Generate elsewhere, run Evaluate + Analyze on the node (recommended)

1. **On your laptop (or anywhere with Ollama/API):**
   ```bash
   python src/generate.py --model qwen2.5-coder:14b --n-samples 5 \
     --api-base http://localhost:11434/v1 --output outputs/generated/ollama.jsonl
   ```
   Copy the repo (including `outputs/generated/*.jsonl`) to the HPC system (e.g. `$HOME` or project space).

2. **On Aurora or Polaris:** From the repo directory:
   ```bash
   # Aurora
   qsub scripts/aurora_job.pbs

   # Polaris
   qsub scripts/polaris_job.pbs
   ```
   The PBS scripts use `#PBS -A dagger`. The job runs Phase 2 and Phase 3 only.

---

## Option B: Run everything on one node (full pipeline)

### Polaris (NVIDIA A100) — full pipeline is practical

Polaris nodes have **NVIDIA A100** GPUs, so **Ollama** (or any CUDA-based inference) can run on the node. You can run Generate + Evaluate + Analyze in one job.

**Option B1: Ollama in your home (or project) on Polaris**

1. In an interactive session or a one-node job, install Ollama (or use a pre-built binary in `$HOME/bin`):
   ```bash
   # Interactive: qsub -I -l select=1:system=polaris -l walltime=1:00:00 -l filesystems=home:eagle -q debug -A dagger
   curl -fsSL https://ollama.com/install.sh | sh
   export PATH=$HOME/bin:$PATH  # if you installed to $HOME/bin
   ollama serve &   # background
   ollama pull qwen2.5-coder:7b
   ```

   **All 4 GPUs + `nohup` (good for interactive Polaris):** from the app directory run:
   ```bash
   cd /path/to/DaggerApps/apps/pass-at-k-study
   bash scripts/start_ollama_4gpu_background.sh "your-model:tag"   # starts serve; pulls that model
   # or: bash scripts/start_ollama_4gpu_background.sh            # serve only, pull later
   ```
   This sets `CUDA_VISIBLE_DEVICES=0,1,2,3`, starts `ollama serve` in the background, waits until the API is up, then optionally runs `ollama pull`. Logs: `/tmp/ollama-$USER.log`.

2. Run the full pipeline in the same session (or in a batch script that starts `ollama serve` in the background):
   ```bash
   cd $HOME/dagger-llm-study   # or REPO_ROOT
   source .venv/bin/activate
   ollama serve &
   sleep 10
   python src/generate.py --model qwen2.5-coder:7b --n-samples 5 \
     --api-base http://localhost:11434/v1 --output outputs/generated/polaris.jsonl
   julia --project=. src/evaluate.jl outputs/generated/polaris.jsonl
   julia --project=. src/analyze.jl outputs/evaluated/evaluated_polaris.jsonl
   ```

**Option B2: Batch job that runs full pipeline (Polaris)**

Use the provided full-pipeline script and set your project:

```bash
qsub scripts/polaris_full_job.pbs
```

That script starts Ollama in the job, runs generate → evaluate → analyze, then exits. It requires Ollama to be available on the node (installed in `$HOME` or via module/container). See `scripts/polaris_full_job.pbs` for the exact sequence.

### Aurora (Intel Max) — full pipeline is harder

Aurora nodes use **Intel Data Center GPU Max** (Ponte Vecchio). Ollama’s Intel GPU support (SYCL) is limited and may not target Ponte Vecchio. Options:

1. **Run Ollama on CPU** on the node (slow but works): install Ollama in `$HOME`, run `ollama serve` and use a small model (e.g. 7B). Then same flow as Polaris (generate → evaluate → analyze).
2. **Use an API** reachable from Aurora (e.g. a service you run elsewhere) and call it from the job; no local inference on Aurora.
3. **Intel stack (advanced):** Use Intel Extension for Transformers / oneAPI to run a model on the Intel Max GPUs; then either expose an OpenAI-compatible endpoint or adapt `generate.py` to call that stack.

For most users, **Option A (generate elsewhere, run eval+analyze on Aurora)** is the simplest.

---

## PBS scripts in this repo

| Script | System  | Use case |
|--------|---------|----------|
| `scripts/aurora_job.pbs`   | Aurora  | Eval + Analyze only (or add your own Phase 1). |
| `scripts/polaris_job.pbs`  | Polaris | Eval + Analyze only (or add your own Phase 1). |
| `scripts/polaris_full_job.pbs` | Polaris | Full pipeline (Generate + Eval + Analyze) using Ollama on the node. |
| `scripts/start_ollama_4gpu_background.sh` | Polaris (etc.) | `nohup ollama serve` with `CUDA_VISIBLE_DEVICES=0,1,2,3`; optional `ollama pull`. |

Before submitting:

- PBS scripts use **`#PBS -A dagger`** (change if your allocation differs).
- Ensure **Python** (venv/conda with `pip install -e .`) and **Julia** (`julia --project=. -e 'using Pkg; Pkg.instantiate()'`) are set up in the job environment.
- Submit from the repo directory, or pass **`REPO_ROOT`**:  
  `qsub -v REPO_ROOT=/path/to/dagger-llm-study scripts/polaris_job.pbs`

---

## Queues (check ALCF docs for current limits)

- **Polaris:** `debug` (1–2 nodes, ~1 hr), `prod`, `preemptable`, etc. Use `select=1:system=polaris` and `filesystems=home:eagle`.
- **Aurora:** `debug`, `capacity`, etc. Use `select=1` and `filesystems=home:flare`.

Monitor jobs: `qstat -u $USER`.
