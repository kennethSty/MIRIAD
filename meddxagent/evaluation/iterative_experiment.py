from typing import Dict, Union
from pathlib import Path
from meddxagent.ddxdriver.utils import Agents
from meddxagent.evaluation.experiment_utils import (
        run_single_experiment, 
        setup_experiment_dir_and_log, 
        setup_toggled_variables, 
        setup_experiment_cfgs
        )
from meddxagent.ddxdriver.logger import log

def iterative_experiment(
        iterative_experiment_folder: Union[str, Path],
        experiment_base_cfg: Dict
        ):
    iterative_experiment_folder, experiment_logging_path = setup_experiment_dir_and_log(
        iterative_experiment_folder
    )

    # Setup dict of variables toggled in experiments
    toggled_variables = setup_toggled_variables(
        toggle_datasets=True,
        toggle_models=True,
        toggle_corpus=True, # toggles rag strategy too 
        toggle_iterations=True,
        experiment_base_cfg=experiment_base_cfg
    )

    # Setup fixed variables during experiment
    fixed_driver_kwargs = {
        "agent_order": [
            Agents.HISTORY_TAKING.value, 
            Agents.RAG.value, 
            Agents.DIAGNOSIS.value
            ],
        "agent_prompt_length": 10,
    }

    fixed_diagnosis_kwargs = {
        "diagnosis_agent_type": "single_llm_standard.SingleLLMStandard",
        "fewshot_num_shots": 0
    }

    fixed_rag_kwargs = {
        "embedding_model_name": "BAAI/bge-large-en-v1.5",
        "top_k_search": 2,
        "max_question_searches": 3,
        "max_keyword_searches": 3,
    }

    cfgs = setup_experiment_cfgs(
        toggled_variables=toggled_variables, 
        variable_order=["corpus_name", "iterations", "dataset", "model"],
        fixed_driver_kwargs = fixed_driver_kwargs,
        fixed_diagnosis_kwargs = fixed_diagnosis_kwargs,        
        fixed_rag_kwargs = fixed_rag_kwargs,
        active_agents={
            Agents.DRIVER.value,
            Agents.DIAGNOSIS.value,
            Agents.RAG.value,
            Agents.HISTORY_TAKING.value
            },
        experiment_base_cfg=experiment_base_cfg
    )

    log.info("Starting to run entire iterative experiment...\n")
    for experiment_number, (
        bench_cfg,
        ddxdriver_cfg,
        diagnosis_cfg,
        history_taking_cfg,
        patient_cfg,
        rag_cfg,
    ) in enumerate(cfgs, start=1):
        run_single_experiment(
            experiment_number=experiment_number,
            ddxdriver_cfg=ddxdriver_cfg,
            history_taking_cfg=history_taking_cfg,
            patient_cfg=patient_cfg,
            diagnosis_cfg=diagnosis_cfg,
            rag_cfg=rag_cfg,
            bench_cfg=bench_cfg,
            experiment_logging_path=experiment_logging_path,
            base_experiment_folder=iterative_experiment_folder,
    )

