# Pass@k Study App

This app packages the full pass@k experiment under `DaggerApps/apps/pass-at-k-study` with a Julia-only pipeline:

- `generate.jl` (LLM generation)
- `evaluate.jl` (framework execution/evaluation)
- `analyze.jl` (pass@k and runtime plots/tables)

The benchmarked frameworks are `dagger`, `iris`, `legate`, and `parsec`. Framework source snippets can be non-Julia, but orchestration/infrastructure is Julia.

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

## Outputs

- `outputs/generated/*.jsonl`: raw generations
- `outputs/evaluated/*.jsonl`: evaluated records
- `tables/pass_at_k.tex`
- `tables/pass_at_k_by_framework.tex`
- `tables/runtime_by_framework.tex`
- `figures/pass_at_k_comparative.pdf`
- `figures/runtime_comparative.pdf`

## Runner command overrides

- `IRIS_BUILD_CMD`, `IRIS_RUN_CMD`
- `LEGATE_RUN_CMD`
- `PARSEC_BUILD_CMD`, `PARSEC_RUN_CMD`

## App structure

- `benchmark/tasks/dagger/*.jl` + `task_utils.jl`
- `benchmark/tasks/{iris,legate,parsec}/tasks.json` + reference source files
- `benchmark/prompts/`
- `src/` (`generate.jl`, `evaluate.jl`, `analyze.jl`, `utils/`)
- `scripts/` (setup, smoke, full run)
