#!/bin/bash
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm.out

cd ~/magisterka/magisterka/dataset_creation

mkdir -p results
mkdir -p logs
source caches_to_tmpdir.sh
MODELS=(
    "google/gemma-4-31b-it"
    "google/gemma-4-8b-it"
    "Qwen/Qwen2.5-72B-Instruct"
    "Qwen/Qwen2.5-Coder-32B-Instruct"
    "mistralai/Mixtral-8x22B-Instruct-v0.1"
    "PrimeIntellect/INTELLECT-1-Instruct"
    "openai/gpt-oss-120b"
    "openai/gpt-oss-20b"
)

# Require single model argument
if [ -z "$1" ]; then
    echo "Usage: sbatch script.sh <model_name>"
    exit 1
fi

MODEL="$1"

echo "====================================="
echo "RUNNING $MODEL"
echo "====================================="

bash run_model.sh "$MODEL" 4