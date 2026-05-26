#!/bin/bash
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm.out
#SBATCH --gres=storage:lustre:1000G

cd ~/magisterka/magisterka/dataset_creation

mkdir -p results
mkdir -p logs
source caches_to_tmpdir.sh


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