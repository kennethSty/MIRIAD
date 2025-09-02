#!/bin/bash

# ----------------------------
# Run Retrieval Agent Experiment in tmux
# ----------------------------

# Default configuration options
NUM_PATIENTS=20
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
SESSION_NAME="rag_experiment"

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
        --session_name)
            SESSION_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--num_patients NUM] [--model_name MODEL] [--session_name NAME]"
            exit 1
            ;;
    esac
done

EXPERIMENT_FOLDER="experiments/rag/rag_${MODEL_NAME}"

echo "Starting tmux session: ${SESSION_NAME}"
echo "Number of patients: ${NUM_PATIENTS}"
echo "Model name: ${MODEL_NAME}"
echo "Experiment folder: ${EXPERIMENT_FOLDER}"

# Determine MIRIAD root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.."
CMD="export PYTHONPATH=${ROOT_DIR}:\$PYTHONPATH && \
     export CUDA_VISIBLE_DEVICES=4,5,6,7 && \
     echo 'Using GPUs: '\$CUDA_VISIBLE_DEVICES && \
     python -m meddxagent.evaluation.experiment \
        --experiment_type rag \
        --experiment_folder ${EXPERIMENT_FOLDER} \
        --num_patients ${NUM_PATIENTS}"

# Kill old session if exists
tmux has-session -t "${SESSION_NAME}" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "Session ${SESSION_NAME} already exists. Killing it..."
    tmux kill-session -t "${SESSION_NAME}"
fi

# Start new tmux session in detached mode
tmux new-session -d -s "${SESSION_NAME}" "${CMD}"

echo "Experiment running in tmux session: ${SESSION_NAME}"
echo "Attach with:   tmux attach -t ${SESSION_NAME}"
echo "Detach with:   Ctrl+b d"
echo "Kill session:  tmux kill-session -t ${SESSION_NAME}"

