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

sleep 120

python benchmark.py

kill $SERVER_PID