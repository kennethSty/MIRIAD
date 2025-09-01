import traceback
import itertools
import time
import yaml 
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Union
from meddxagent.ddxdriver.utils import Agents
from meddxagent.ddxdriver.rag_agents._searchrag_utils import Corpus
from meddxagent.ddxdriver.run_ddxdriver import run_ddxdriver
from meddxagent.ddxdriver.logger import log, enable_logging, set_file_handler, log_json_data
MEDDX_ROOT = Path(__file__).resolve().parent.parent

def get_experiment_paths(args: Namespace, experiment_cfg):
    active_models = experiment_cfg["active_models"]
    model_name = "_".join(
        m["model_name"].split("/")[-1].lower() for m in active_models
    )
    base_folders = {
        "diagnosis": MEDDX_ROOT / f"experiments/diagnosis/diagnosis_{model_name}",
        "history_taking": MEDDX_ROOT / f"experiments/history_taking/history_taking_{model_name}",
        "rag": MEDDX_ROOT / f"experiments/rag/rag_{model_name}",
        "iterative": MEDDX_ROOT / f"experiments/iterative/iterative_{model_name}",
    }

    if args.experiment_type not in base_folders.keys():
        raise ValueError (
                "experiment type has to be one of the following: \n" + \
                "'all', 'diagnosis', 'history_taking', 'rag', 'iterative'")
    elif args.experiment_type != "all":
        return {args.experiment_type: base_folders[args.experiment_type]}
    else:
        return base_folders

def setup_experiment_cli_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run MEDDxAgent experiments")
    parser.add_argument(
        "--experiment_type",
        choices=["diagnosis", "history_taking", "rag", "iterative", "all"],
        default="all",
        help="Type of experiment to run"
    )
    parser.add_argument(
        "--experiment_folder",
        type=str,
        help="Custom experiment folder path (optional, will auto-generate if not provided)"
    )
    parser.add_argument(
        "--num_patients",
        type=int,
        help=f"Number of patients to run experiments on"
    )
    return parser

def setup_experiment_dir_and_log(experiment_folder: Union[str, Path]):
    experiment_folder = Path(experiment_folder)
    experiment_logging_path = experiment_folder / "experiment_logs.log"
    set_file_handler(experiment_logging_path, mode="a")
    enable_logging(console_logging=True, file_logging=True)
    
    return experiment_folder, experiment_logging_path

def setup_toggled_variables(
        experiment_base_cfg: Dict,
        toggle_models = False, 
        toggle_datasets = False,
        toggle_max_questions = False,
        toggle_max_question_searches = False,
        toggle_corpus = False,
        toggle_diagnosis_llm_type = False,
        toggle_fewshot_type=False,
        toggle_num_shots=False,
        toggle_max_keywords = False,
        toggle_top_k_search = False,
        toggle_iterations = False
        ) -> Dict:
    variable_dict = {}
    if toggle_models:
        variable_dict["model"] = experiment_base_cfg["active_models"]
    if toggle_datasets:
        variable_dict["dataset"] = ["ddxplus.DDxPlus", "icraftmd.ICraftMD"]
    if toggle_max_questions:
        variable_dict["max_questions"] = [5, 10, 15]
    if toggle_corpus:
        variable_dict["corpus_name"] = [Corpus.PUBMED, Corpus.WIKIPEDIA]
    if toggle_fewshot_type:
        variable_dict["fewshot_type"] = ["static", "dynamic", "none"]
    if toggle_num_shots:
        variable_dict["num_shots"] = [0,5]
    if toggle_diagnosis_llm_type:
        variable_dict["diagnosis_agent_type"] = [
            "single_llm_cot.SingleLLMCOT", 
            "single_llm_standard.SingleLLMStandard"
        ]
    if toggle_max_question_searches:
        variable_dict["max_question_searches"] = [3, 4]
    if toggle_max_keywords:
        variable_dict["max_keyword_searches"] = [3, 4]
    if toggle_top_k_search:
        variable_dict["top_k_search"] = [2, 3]
    if toggle_iterations:
        variable_dict["iterations"] = [2, 3]
    return variable_dict

def setup_benchmark_cfg(dataset_name: str, num_patients: int, enforce_diagnosis_options = True) -> Dict:
    bench_cfg = yaml.safe_load((MEDDX_ROOT / "configs/bench.yml").read_text())
    bench_cfg["class_name"] = f"meddxagent.ddxdriver.benchmarks.{dataset_name}"
    bench_cfg["num_patients"] = num_patients
    bench_cfg["enforce_diagnosis_options"] = enforce_diagnosis_options
    return bench_cfg

def get_driver_cfg(
    model_name: str,
    model_class_name: str,
    agent_order: List[str],
    driver_type = "fixed_choice.FixedChoice", 
    agent_prompt_length = 0,
    iterations=1,
    only_patient_initial_information=False
    ) -> Dict: 

    return  {
        "class_name": f"meddxagent.ddxdriver.ddxdrivers.{driver_type}",
        "config": {
            "seed": 42,
            "agent_prompt_length": agent_prompt_length,
            "agent_order": agent_order,
            "iterations": iterations,
            "only_patient_initial_information": only_patient_initial_information,
            "model": {
                "class_name": f"meddxagent.ddxdriver.models.{model_class_name}",
                "config": {"model_name": model_name},
            },
        },
    }

def get_diagnosis_cfg(
    model_name: str,
    model_class_name: str,
    fewshot_type = "none",
    fewshot_num_shots = 0,
    diagnosis_agent_type = "single_llm_standard.SingleLLMStandard" 
    ) -> Dict:
    return {
        "class_name": f"meddxagent.ddxdriver.diagnosis_agents.{diagnosis_agent_type}",
        "config": {
            "model": {
                "class_name": f"meddxagent.ddxdriver.models.{model_class_name}",
                "config": {"model_name": model_name},
            },
            "fewshot": {"type": fewshot_type, "num_shots": fewshot_num_shots},
        },
    }

def get_history_taking_cfg(
    model_name: str,
    model_class_name: str,
    max_questions: int,
    history_taking_agent_type = "llm_history_taking"
    ) -> Dict:
    history_taking_cfg = yaml.safe_load(
        (MEDDX_ROOT / f"configs/history_taking_agents/{history_taking_agent_type}.yml").read_text()
    )
    history_taking_cfg["config"]["max_questions"] = max_questions
    history_taking_cfg["config"]["model"] = {
        "class_name": f"meddxagent.ddxdriver.models.{model_class_name}",
        "config": {"model_name": model_name},
    }
    return history_taking_cfg

def get_miriad_rag_cfg(
    emb_model_name: str = "BAAI/bge-large-en-v1.5",
    content: str = "qa",
    llm_class_name: str = "llama31_8b.Llama318B",
    llm_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    host: str = "localhost",
    port: int = 6333,
    top_k_search: int = 2,
    max_question_searches: int = 3,
    **kwargs
) -> dict:
    return {
        "class_name": "meddxagent.ddxdriver.rag_agents.miriad_rag.MiriadRAG",
        "config": {
            "embedding": {
                "model_names": [
                    "BAAI/bge-large-en-v1.5",
                    "sentence-transformers/all-MiniLM-L6-v2",
                ],
                "contents": ["qa"],
                "emb_model_name": emb_model_name,
                "content": content,
            },
            "model": {
                "class_name": f"meddxagent.ddxdriver.models.{llm_class_name}",
                "config": {"model_name": llm_model_name},
            },
            "qdrant": {
                "host": host,
                "port": port,
                "collection": f"miriad_{emb_model_name}_{content}",
            },
            "retrieval": {
                "top_k_search": top_k_search,
                "max_question_searches": max_question_searches,
            },
        },
    }

def get_search_rag_cfg(
    corpus_name: Corpus,
    model_class_name: str,
    model_name: str,
    top_k_search: int = 2,
    max_keyword_searches: int = 3,
    **kwargs
) -> dict:
    return {
        "class_name": "meddxagent.ddxdriver.rag_agents.searchrag_standard.SearchRAGStandard",
        "config": {
            "corpus_name": corpus_name.value,
            "top_k_search": top_k_search,
            "max_keyword_searches": max_keyword_searches,
            "model": {
                "class_name": f"meddxagent.ddxdriver.models.{model_class_name}",
                "config": {"model_name": model_name},
            },
        },
    }


def get_patient_cfg(
    model_name: str,
    model_class_name: str,
    patient_agent_type = "llm_patient"
    ) -> Dict:
    patient_cfg = yaml.safe_load(
        (MEDDX_ROOT / f"configs/patient_agents/{patient_agent_type}.yml").read_text()
    )
    patient_cfg["config"]["model"] = {
        "class_name": f"meddxagent.ddxdriver.models.{model_class_name}",
        "config": {"model_name": model_name},
    }
    return patient_cfg

def setup_experiment_cfgs(
        toggled_variables, 
        variable_order, 
        experiment_base_cfg,
        fixed_history_taking_kwargs = {},
        fixed_patient_kwargs = {},
        fixed_driver_kwargs = {},
        fixed_diagnosis_kwargs = {},
        fixed_rag_kwargs = {},
        active_agents = {Agents.DRIVER.value, Agents.DIAGNOSIS.value}
        ) -> List[Dict]:
    if len(toggled_variables.keys()) != len(variable_order):
        raise ValueError(f"expected {len(toggled_variables.keys())}, got {len(variable_order)}")
    cfgs = []
    for combination in itertools.product(*(toggled_variables[k] for k in variable_order)):
        #Map each variable name to its current value
        var_values = {k: v for k, v in zip(variable_order, combination)}
        try:
            current_cfgs = []
            bench_cfg = setup_benchmark_cfg(
                dataset_name=var_values.get("dataset", experiment_base_cfg["dataset"]), 
                num_patients=experiment_base_cfg["num_patients"], 
                enforce_diagnosis_options=True
            )
            current_cfgs.append(bench_cfg)
            if Agents.DRIVER.value in active_agents:
                model = var_values.get("model", experiment_base_cfg["active_models"][0])
                ddxdriver_cfg = get_driver_cfg(
                    model_class_name=model["class_name"],
                    model_name=model["model_name"],
                    **fixed_driver_kwargs
                ) 
                current_cfgs.append(ddxdriver_cfg)
            if Agents.DIAGNOSIS.value in active_agents:
                model = var_values.get("model", experiment_base_cfg["active_models"][0])
                diagnosis_cfg = get_diagnosis_cfg(
                    model_class_name=model["class_name"],
                    model_name=model["model_name"],
                    **fixed_diagnosis_kwargs
                )
                current_cfgs.append(diagnosis_cfg)
            if Agents.HISTORY_TAKING.value in active_agents:
                # history taking and patient agent operate together
                model = var_values.get("model", experiment_base_cfg["active_models"][0])
                history_taking_cfg = get_history_taking_cfg(
                    model_name=model["model_name"],
                    model_class_name=model["class_name"],
                    max_questions=var_values.get("max_questions", 15),
                    **{k: v for k, v in fixed_history_taking_kwargs
                       if k != "max_questions"}
                )
                patient_cfg = get_patient_cfg(
                    model_name=model["model_name"],
                    model_class_name=model["class_name"],
                    **fixed_patient_kwargs
                )
                current_cfgs.extend([history_taking_cfg, patient_cfg])

            if Agents.RAG.value in active_agents:
                # Search rag uses pubmed or wikipedia, while miriad rag uses only miriad 
                model = var_values.get("model", experiment_base_cfg["active_models"][0])
                if var_values.get("corpus_name") == Corpus.MIRIAD:
                    rag_cfg = get_miriad_rag_cfg(
                        emb_model_name=var_values.get("emb_model_name", "BAAI/bge-large-en-v1.5"),
                        content=var_values.get("content", "qa"),
                        **fixed_rag_kwargs
                    )
                else:
                    rag_cfg = get_search_rag_cfg(
                        corpus_name=var_values.get("corpus_name", Corpus.PUBMED),
                        model_class_name=model["class_name"],
                        model_name=model["model_name"],
                        **fixed_rag_kwargs
                    )
                current_cfgs.append(rag_cfg)
            cfgs.append(tuple(current_cfgs))
            current_cfgs = []
        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"Error with setting up diagnosis experiments:\n{e}\nTraceback:\n{tb}")

    num_combinations = len(list(itertools.product(*toggled_variables.values())))
    if len(cfgs) != num_combinations:
        log.error(f"Did not correctly generate {num_combinations} pairs\n")
        raise RuntimeError(f"Expected {num_combinations}, got {len(cfgs)}")
    else:
        log.info(f"Correctly generated {num_combinations} different experiments\n")
    return cfgs

def run_single_experiment(
    experiment_number: int,
    ddxdriver_cfg: Dict,
    diagnosis_cfg: Dict,
    bench_cfg: Dict,
    experiment_logging_path: Path,
    base_experiment_folder: Path,
    history_taking_cfg = None,
    patient_cfg = None,
    rag_cfg = None,
):
    """Run one experiment iteration with logging and error handling."""
    set_file_handler(experiment_logging_path, mode="a")
    log.info(f"STARTING EXPERIMENT {experiment_number}...\n")
    start_time = time.time()

    try:
        experiment_folder = base_experiment_folder / str(experiment_number)

        # Log config
        json_data = {
            "meddxagent.ddxdriver.cfg": ddxdriver_cfg,
            "history_taking_cfg": history_taking_cfg,
            "bench_cfg": bench_cfg,
            "patient_cfg": patient_cfg,
            "diagnosis_cfg": diagnosis_cfg,
        }
        log_json_data(json_data=json_data, file_path=experiment_folder / "configs.json")

        # Run driver
        run_ddxdriver(
            bench_cfg=bench_cfg,
            ddxdriver_cfg=ddxdriver_cfg,
            diagnosis_agent_cfg=diagnosis_cfg,
            rag_agent_cfg=rag_cfg,
            history_taking_agent_cfg=history_taking_cfg,
            patient_agent_cfg=patient_cfg,
            experiment_folder=experiment_folder,
        )

        set_file_handler(experiment_logging_path, mode="a")
        log.info(f"FINISHED EXPERIMENT {experiment_number}.\n")
    except Exception as e:
        set_file_handler(experiment_logging_path, mode="a")
        tb = traceback.format_exc()
        log.error(
            f"Error with running experiment {experiment_number}:\n{e}\nTraceback:\n{tb}"
        )

    log.info(
        f"Finished running experiment {experiment_number} in {time.time()-start_time} seconds.\n"
    )
