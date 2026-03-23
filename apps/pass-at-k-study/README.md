# Pass@k Study App

This app packages the full pass@k experiment under `DaggerApps/apps/pass-at-k-study` with a Julia-only pipeline:

- `generate.jl` (LLM generation)
- `evaluate.jl` (framework execution/evaluation)
- `analyze.jl` (pass@k and runtime plots/tables)

The benchmarked frameworks are `dagger`, `iris`, and `legate`. Framework source snippets can be non-Julia, but orchestration/infrastructure is Julia.

## Quick start

From the `DaggerApps` repo root:

```bash
julia --project=apps/pass-at-k-study -e 'using Pkg; Pkg.instantiate()'
bash apps/pass-at-k-study/scripts/run_smoke_test.sh
```

Run the full pipeline:

```bash
bash apps/pass-at-k-study/scripts/run_full_study.sh
```

Ollama (starts `ollama serve` in the background with `nohup` if nothing is listening, then runs the same pipeline; default model `deepseek-coder:33b`):

```bash
bash apps/pass-at-k-study/scripts/run_full_study_ollama.sh
# or: bash apps/pass-at-k-study/scripts/run_full_study_ollama.sh "deepseek-coder:33b" 5
```

Start only Ollama in the background with **four GPUs** visible (`CUDA_VISIBLE_DEVICES=0,1,2,3`), then pull a model:

```bash
cd apps/pass-at-k-study
bash scripts/start_ollama_4gpu_background.sh "deepseek-coder:33b"
```

## Offline docs-as-context (no internet at inference time)

Models in Ollama do not browse the web. To provide documentation context offline, build local excerpts and inject them in prompts:

```bash
cd /eagle/dagger/paper/DaggerApps/apps/pass-at-k-study
julia --project=. scripts/build_docs_context.jl
julia --project=. src/generate.jl --model deepseek-coder:33b --api-base http://127.0.0.1:11434/v1 --docs-context-dir benchmark/prompts/docs_context --docs-top-k 3
```

Useful flags:
- `--docs-context-dir <path>`: location of local docs snippets/index.
- `--docs-top-k <int>`: max snippets injected per task.
- `--disable-docs-context`: run generation without local docs injection (A/B baseline).

Generation reads only local files during prompt construction; no live internet access is required once context files are prepared.

## Outputs

- `outputs/generated/*.jsonl`: raw generations
- `outputs/evaluated/*.jsonl`: evaluated records
- `tables/pass_at_k.md`
- `tables/pass_at_k_by_framework.md`
- `tables/runtime_by_framework.md`
- `figures/pass_at_k_by_task.png`
- `figures/pass_at_k_comparative.png`
- `figures/runtime_comparative.png`

## Runner command overrides

See **`docs/RUNNER_ENV.md`** for Iris / Legate setup. On this paper checkout, **`source scripts/env_paper_runtime.sh`** sets Iris paths to **`$PAPER_ROOT/.runtime/opt/iris`** (if present).

- **Iris:** `IRIS_BUILD_CMD`, `IRIS_RUN_CMD`
- **Legate:** `LEGATE_RUN_CMD` and/or `LEGATE_PYTHON` (after `scripts/setup_legate_conda.sh`)

## App structure

- `benchmark/tasks/dagger/*.jl` + `task_utils.jl`
- `benchmark/tasks/{iris,legate}/tasks.json` + reference source files
- `benchmark/prompts/`
- `src/` (`generate.jl`, `evaluate.jl`, `analyze.jl`, `utils/`)
- `scripts/` (setup, smoke, full run)
