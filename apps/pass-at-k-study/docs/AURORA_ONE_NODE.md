# Running the study on one node of Aurora (ALCF)

**See [ALCF_ONE_NODE.md](ALCF_ONE_NODE.md) for the full guide covering both Aurora and Polaris.**

Aurora uses **PBS Pro** (not Slurm). One node has **6 GPU tiles** (12 Intel Data Center GPU Max / Ponte Vecchio). This doc covers running the pipeline on a single node.

## Option A: Generate elsewhere, run Evaluate + Analyze on Aurora (recommended)

1. **On your laptop (or another machine with Ollama/API):**  
   Run Phase 1 and upload the generated JSONL to Aurora:
   ```bash
   python src/generate.py --model qwen2.5-coder:14b --n-samples 5 \
     --api-base http://localhost:11434/v1 --output outputs/generated/ollama.jsonl
   ```
   Then copy the repo (including `outputs/generated/ollama.jsonl`) to Aurora, e.g. to `$HOME` or `$WORK` (flare).

2. **On Aurora:**  
   Request one node and run only Phase 2 and Phase 3:
   ```bash
   # In your repo on Aurora (e.g. $WORK/dagger-llm-study or $HOME/dagger-llm-study)
   qsub -v REPO_ROOT="$(pwd)" scripts/aurora_job.pbs
   ```
   Or edit `scripts/aurora_job.pbs`: set `#PBS -A PROJECT_ALLOCATION` to your ALCF project, then:
   ```bash
   qsub scripts/aurora_job.pbs
   ```
   The script will use the latest `outputs/generated/*.jsonl`, run `evaluate.jl`, then `analyze.jl`, and write to `outputs/evaluated/`, `figures/`, and `tables/`.

**Why this works:** Phase 1 needs an LLM API or local inference (Ollama, etc.). Aurora may not have Ollama installed; running generation on your laptop and copying results avoids that. Phase 2 (evaluate) runs the generated Julia code; with `task_utils.jl` and oneAPI, Dagger can use the node’s Intel GPUs. Phase 3 (analyze) is pure stats/plotting.

---

## Option B: Full pipeline on one node (Generate + Evaluate + Analyze)

To run **generation on Aurora** you need an LLM backend on the node. Options:

1. **Ollama (if you build/install it in your job or home):**  
   - Build Ollama for Linux and a model that fits the node’s memory.  
   - In the job: start `ollama serve`, then run `generate.py` with `--api-base http://localhost:11434/v1`.  
   - Set `GENERATE_ON_NODE=1` in the job and add the actual `generate.py` and `ollama serve` commands to `aurora_job.pbs` (or a wrapper script).

2. **LLM API reachable from the node:**  
   - If you have an internal API (e.g. served elsewhere and reachable from Aurora), set `OPENAI_API_KEY` and `API_BASE` (or equivalent) and run `generate.py` in the job the same way as on your laptop.

3. **Intel Extension for Transformers / llama.cpp (Intel GPU):**  
   - Use oneAPI and a Python or C++ inference stack that runs on Intel Max GPUs, then either call it via a small local API or adapt `generate.py` to call that stack instead of OpenAI/Ollama.

The provided `aurora_job.pbs` does **not** start Ollama or any inference by default; it only runs Phase 2 and Phase 3 unless you set `GENERATE_ON_NODE=1` and add your own Phase 1 commands.

---

## Setup on Aurora

- **Python:** Use the system Python or a conda env (e.g. `module load conda` if available; or create a venv and `pip install -e .` in the repo).
- **Julia:** Install Julia (e.g. in your home or project dir), then in the repo run:
  ```bash
  julia --project=. -e 'using Pkg; Pkg.instantiate()'
  ```
- **Paths:** The PBS script uses `PBS_O_WORKDIR` if set (directory from which you ran `qsub`); otherwise it uses `$HOME/dagger-llm-study`. Set `REPO_ROOT` in the job if your repo is elsewhere (e.g. `$WORK/dagger-llm-study`).

---

## PBS details (ALCF Aurora)

- **Scheduler:** PBS Pro.  
- **One node:** `#PBS -l select=1`.  
- **Queues:** e.g. `debug` (short, 1–2 nodes), `capacity` (longer, multi-node). Use the queue names and limits from the current [ALCF Aurora Running Jobs](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/) documentation.  
- **Filesystems:** Request the filesystems where your repo and data live (e.g. `home:flare`).  
- **Project:** Replace `PROJECT_ALLOCATION` in `aurora_job.pbs` with your ALCF project code.

Submit from the directory that contains your repo (or set `REPO_ROOT`):

```bash
qsub scripts/aurora_job.pbs
```

Monitor:

```bash
qstat -u $USER
```

Output and errors go to the file(s) specified by `#PBS -o` (e.g. `dagger_study_<jobid>.out`).
