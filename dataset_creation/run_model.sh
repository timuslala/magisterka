#!/bin/bash
set -euo pipefail  # Exit on error, undefined vars, pipe failures

MODEL=$1
TP=$2
export HF_HOME=$TMPDIR
export TRANSFORMERS_CACHE=$TMPDIR
export HUGGINGFACE_HUB_CACHE=$TMPDIR
export HF_HUB_CACHE=$TMPDIR

echo "STARTING MODEL: $MODEL"
cd ~/magisterka/magisterka/dataset_creation

# Clean up old log
mkdir -p logs
> logs/vllm.log

uv run python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --tensor-parallel-size $TP \
    --download-dir $TMPDIR/hf_cache/hub \
    --max-model-len 16384 \
    --host 0.0.0.0 \
    --port 8000 \
    --chat-template-content-format string \
    --trust-remote-code \
    > logs/vllm.log 2>&1 &

SERVER_PID=$!

echo "Waiting for server..."

for i in {1..120}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "Server ready after $((i * 10)) seconds"
        break
    fi
    echo "not ready yet... $i"
    sleep 10
done

# Test with a minimal request first
echo "Testing with minimal request..."
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"max_tokens\": 10}" || true

echo "Running benchmark..."
uv run benchmark.py $MODEL || echo "BENCHMARK EXITED WITH CODE: $?"

# Check if server still alive
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server still running, killing..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null || true
else
    echo "Server already dead!"
fi