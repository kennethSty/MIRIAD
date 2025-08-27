#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent directory of the script directory)
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

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

EXPERIMENT_FOLDER="experiments/diagnosis/diagnosis_${MODEL_NAME}"

echo "Running diagnosis agent experiment..."
echo "Number of patients: ${NUM_PATIENTS}"
echo "Model name: ${MODEL_NAME}"
echo "Experiment folder: ${EXPERIMENT_FOLDER}"

# Add project root to Python path and run from the scripts directory
cd "${SCRIPT_DIR}" && PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}" python experiment.py --experiment_type diagnosis --experiment_folder "${EXPERIMENT_FOLDER}" --num_patients "${NUM_PATIENTS}"
