

# Determine MIRIAD root directory
# Consequence: import relative to MIRIAD
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.."
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
echo "Root dir: ${ROOT_DIR}"

CUDA_VISIBLE_DEVICES=5 

# Assumes script to be run from scripts diectory
python -m meddxagent.ddxdriver.run_ddxdriver \
    --bench_cfg "../configs/bench.yml" \
    --ddxdriver_cfg "../configs/ddx_drivers/fixed_choice.yml" \
    --diagnosis_agent_cfg "../configs/diagnosis_agents/single_llm_standard.yml"\
    --history_taking_agent_cfg "../configs/history_taking_agents/llm_history_taking.yml"\
    --patient_agent_cfg "../configs/patient_agents/llm_patient.yml" \
    --rag_agent_cfg "../configs/rag_agents/miriadrag.yml"
