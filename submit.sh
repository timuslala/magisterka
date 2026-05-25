#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm.out

module load cuda
module load miniconda

conda activate llm

mkdir -p results
mkdir -p logs

MODELS=(
    "microsoft/codebert-base"
    "google/gemma-4-31b-it"
    "google/gemma-4-8b-it"
    "Qwen/Qwen2.5-72B-Instruct"
    "Qwen/Qwen2.5-Coder-32B-Instruct"
    "mistralai/Mixtral-8x22B-Instruct-v0.1"
    "PrimeIntellect/INTELLECT-1-Instruct"
    "openai/gpt-oss-120b"
    "openai/gpt-oss-20b"
)

for MODEL in "${MODELS[@]}"
do
    echo "====================================="
    echo "RUNNING $MODEL"
    echo "====================================="

    bash run_model.sh $MODEL 4

    sleep 30
done