#!/bin/bash

MODEL=$1
TP=$2

echo "STARTING MODEL: $MODEL"

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --tensor-parallel-size $TP \
    --host 0.0.0.0 \
    --port 8000 \
    > logs/vllm.log 2>&1 &

SERVER_PID=$!

echo "Waiting for server..."

echo "Waiting for vLLM to be ready..."

for i in {1..120}; do
    curl -s http://localhost:8000/v1/models && break
    echo "not ready yet... $i"
    sleep 10
done

python benchmark.py $MODEL

kill $SERVER_PID