#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default configuration options (no environment variable dependencies)
NUM_PATIENTS=100
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_patients)
            NUM_PATIENTS="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--num_patients NUM] [--model_name MODEL]"
            exit 1
            ;;
    esac
done

EXPERIMENT_FOLDER="experiments/iterative/iterative_${MODEL_NAME}"

echo "Running iterative experiment..."
echo "Number of patients: ${NUM_PATIENTS}"
echo "Model name: ${MODEL_NAME}"
echo "Experiment folder: ${EXPERIMENT_FOLDER}"

# Determine MIRIAD root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.."
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
echo "Root dir: ${ROOT_DIR}"

# Optional: Set which devises to use 
export CUDA_VISIBLE_DEVICES=4,5,6,7
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}"

python -m meddxagent.evaluation.experiment \
	--experiment_type iterative \
	--experiment_folder "${EXPERIMENT_FOLDER}" \
       	--num_patients "${NUM_PATIENTS}" 
