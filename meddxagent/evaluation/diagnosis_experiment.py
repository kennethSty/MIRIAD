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

def diagnosis_experiment(
        experiment_folder: Union[str, Path],
        experiment_base_cfg: Dict
        ):
    experiment_folder, experiment_logging_path = setup_experiment_dir_and_log(
        experiment_folder
    )

    # Setup dict of variables toggled in experiments
    toggled_variables = setup_toggled_variables(
        toggle_datasets=True,
        toggle_models=True,
        toggle_diagnosis_llm_type=True,
        toggle_num_shots=True,
        toggle_fewshot_type=True,
        experiment_base_cfg=experiment_base_cfg
    )

    # Setup fixed variables during experiment
    fixed_driver_kwargs = {
        "agent_order": [Agents.DIAGNOSIS.value],
        "agent_prompt_length": 0,
        "iterations": 1,
        "only_patient_initial_information": True,
    }

    cfgs = setup_experiment_cfgs(
        toggled_variables=toggled_variables, 
        variable_order=["dataset", "model", "num_shots", "fewshot_type", "diagnosis_agent_type"],
        fixed_driver_kwargs = fixed_driver_kwargs,
        active_agents={
            Agents.DRIVER.value,
            Agents.DIAGNOSIS.value,
            },
        experiment_base_cfg=experiment_base_cfg
    )

    log.info("Starting to run entire history taking experiment...\n")
    for experiment_number, (
        bench_cfg,
        ddxdriver_cfg,
        diagnosis_cfg,
    ) in enumerate(cfgs, start=1):
        run_single_experiment(
            experiment_number=experiment_number,
            ddxdriver_cfg=ddxdriver_cfg,
            diagnosis_cfg=diagnosis_cfg,
            bench_cfg=bench_cfg,
            experiment_logging_path=experiment_logging_path,
            base_experiment_folder=experiment_folder,
    )

