import yaml

from meddxagent.evaluation.diagnosis_experiment import diagnosis_experiment
from meddxagent.evaluation.iterative_experiment import iterative_experiment
from meddxagent.evaluation.history_taking_experiment import history_taking_experiment
from meddxagent.evaluation.rag_experiment import rag_experiment
from meddxagent.evaluation.experiment_utils import MEDDX_ROOT, setup_experiment_cli_parser, get_experiment_paths


def main():
    experiment_base_cfg = yaml.safe_load((MEDDX_ROOT / "configs/experiments/run_experiment.yml").read_text())
    parser = setup_experiment_cli_parser()
    args = parser.parse_args()
    experiment_paths = get_experiment_paths(
            args, 
            experiment_base_cfg
            )

    # Run specific experiment type
    if args.experiment_type == "diagnosis":
        diagnosis_experiment(
            experiment_paths["diagnosis"], 
            experiment_base_cfg
            )
    
    elif args.experiment_type == "history_taking":
        history_taking_experiment(
            experiment_paths["history_taking"], 
            experiment_base_cfg
            )
    
    elif args.experiment_type == "rag":
        rag_experiment(
            experiment_paths["rag"], 
            experiment_base_cfg
            )
    
    elif args.experiment_type == "iterative":
        iterative_experiment(
            experiment_paths["iterative"], 
            experiment_base_cfg
            )
    
    elif args.experiment_type == "all":
        # Run all experiment types with current models (original behavior)
        diagnosis_experiment(
            experiment_paths["diagnosis"], 
            experiment_base_cfg
            )
        history_taking_experiment(
            experiment_paths["history_taking"], 
            experiment_base_cfg
            )
        rag_experiment(
            experiment_paths["rag"], 
            experiment_base_cfg
            )
        iterative_experiment(
            experiment_paths["iterative"], 
            experiment_base_cfg
            )


if __name__ == "__main__":
    main()
