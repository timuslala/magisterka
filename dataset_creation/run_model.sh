#!/bin/bash

MODEL=$1
TP=$2
export HF_HOME=$TMPDIR
export TRANSFORMERS_CACHE=$TMPDIR
export HUGGINGFACE_HUB_CACHE=$TMPDIR
export HF_HUB_CACHE=$TMPDIR
echo "STARTING MODEL: $MODEL"
cd ~/magisterka/magisterka/dataset_creation
uv run python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --tensor-parallel-size $TP \
    --download-dir $TMPDIR/hf_cache/hub \
    --disable-custom-all-reduce \
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

uv run benchmark.py $MODEL

kill $SERVER_PID