#!/usr/bin/env bash
# End-to-end quick test for pass-at-k-study.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Smoke test ==="
echo "Phase 1: Generate (1 task, 2 samples)..."
# Prefer: OpenAI (if key set) -> Ollama (free, local) -> static fallback
SMOKE_GENERATED=""
if julia --project=. -e '
  using HTTP, JSON3
  r = HTTP.post("https://api.openai.com/v1/chat/completions",
    ["Content-Type" => "application/json", "Authorization" => "Bearer " * get(ENV, "OPENAI_API_KEY", "")],
    JSON3.write(Dict("model"=>"gpt-4o-mini","messages"=>[Dict("role"=>"user","content"=>"hi")],"max_tokens"=>2)))
  exit(r.status == 200 ? 0 : 1)
' 2>/dev/null; then
  echo "Using OpenAI gpt-4o-mini"
  julia --project=. src/generate.jl --model gpt-4o-mini --n-samples 2 --task l1_basic_spawn --output outputs/generated/smoke.jsonl && SMOKE_GENERATED=1
fi
if [ -z "$SMOKE_GENERATED" ] && curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Using Ollama (free local model). Start with: ollama serve && ollama pull qwen2.5-coder:7b"
  OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5-coder:7b}"
  julia --project=. src/generate.jl --model "$OLLAMA_MODEL" --n-samples 2 --task l1_basic_spawn \
    --api-base http://localhost:11434/v1 --output outputs/generated/smoke.jsonl && SMOKE_GENERATED=1
fi
if [ -z "$SMOKE_GENERATED" ]; then
  echo "No API (OpenAI or Ollama); writing minimal smoke JSONL."
  mkdir -p outputs/generated
  printf '%s\n' '{"task":"l1_basic_spawn","task_file":"l1_basic_spawn.jl","model":"smoke","sample_id":0,"framework":"dagger","response":"```julia\nfunction parallel_sum(arrays::Vector{Matrix{Float32}})\n  tasks = [Dagger.@spawn scope=GPU_SCOPES[i] sum(arrays[i]) for i in 1:4]\n  return Float32[fetch(t) for t in tasks]\nend\n```"}' \
          '{"task":"l1_basic_spawn","task_file":"l1_basic_spawn.jl","model":"smoke","sample_id":1,"framework":"dagger","response":"```julia\nfunction parallel_sum(arrays::Vector{Matrix{Float32}})\n  return Float32[sum(arrays[i]) for i in 1:4]\nend\n```"}' \
          > outputs/generated/smoke.jsonl
fi

echo "Phase 2: Evaluate..."
julia --project=. src/evaluate.jl outputs/generated/smoke.jsonl

echo "Phase 3: Analyze..."
julia --project=. src/analyze.jl outputs/evaluated/evaluated_smoke.jsonl

echo "=== Smoke test done ==="
