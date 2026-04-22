#!/usr/bin/env bash
# Reduced-tier pass@k: same pipeline as `run_full_study.sh`; phase 1 may use APIs
# or a static JSONL fallback, then always phases 2–3 via the shared driver.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PASSK_TASK="${PASSK_TASK:-l1_basic_spawn}"
export PASSK_OUTPUT="${PASSK_OUTPUT:-outputs/generated/smoke.jsonl}"
mkdir -p "$(dirname "$PASSK_OUTPUT")"

echo "=== Smoke test (pass@k, reduced parameters) ==="

SMOKE_DONE=""
if julia --project=. -e '
  using HTTP, JSON3
  r = HTTP.post("https://api.openai.com/v1/chat/completions",
    ["Content-Type" => "application/json", "Authorization" => "Bearer " * get(ENV, "OPENAI_API_KEY", "")],
    JSON3.write(Dict("model"=>"gpt-4o-mini","messages"=>[Dict("role"=>"user","content"=>"hi")],"max_tokens"=>2)))
  exit(r.status == 200 ? 0 : 1)
' 2>/dev/null; then
  echo "Using OpenAI gpt-4o-mini"
  unset PASSK_SKIP_GENERATE
  bash "$SCRIPT_DIR/run_full_study.sh" gpt-4o-mini 2
  SMOKE_DONE=1
fi

if [ -z "$SMOKE_DONE" ] && curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Using Ollama (local). Start with: ollama serve && ollama pull qwen2.5-coder:7b"
  OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5-coder:7b}"
  export API_BASE="http://localhost:11434/v1"
  unset PASSK_SKIP_GENERATE
  bash "$SCRIPT_DIR/run_full_study.sh" "$OLLAMA_MODEL" 2
  SMOKE_DONE=1
fi

if [ -z "$SMOKE_DONE" ]; then
  echo "No API (OpenAI or Ollama); writing minimal smoke JSONL, then evaluate/analyze via run_full_study.sh."
  mkdir -p outputs/generated
  printf '%s\n' '{"task":"l1_basic_spawn","task_file":"l1_basic_spawn.jl","model":"smoke","sample_id":0,"framework":"dagger","response":"```julia\nfunction parallel_sum(arrays::Vector{Matrix{Float32}})\n  tasks = [Dagger.@spawn scope=GPU_SCOPES[i] sum(arrays[i]) for i in 1:4]\n  return Float32[fetch(t) for t in tasks]\nend\n```"}' \
          '{"task":"l1_basic_spawn","task_file":"l1_basic_spawn.jl","model":"smoke","sample_id":1,"framework":"dagger","response":"```julia\nfunction parallel_sum(arrays::Vector{Matrix{Float32}})\n  return Float32[sum(arrays[i]) for i in 1:4]\nend\n```"}' \
          > "$PASSK_OUTPUT"
  export PASSK_SKIP_GENERATE=1
  export PASSK_GENERATED="$REPO_ROOT/$PASSK_OUTPUT"
  bash "$SCRIPT_DIR/run_full_study.sh" smoke 2
fi

echo "=== Smoke test done ==="
