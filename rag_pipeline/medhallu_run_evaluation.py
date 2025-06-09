import os
import time
import datetime
import fire
import pickle
import toml
import yaml
from termcolor import colored
from datasets import load_dataset
from tqdm import tqdm
from src.printutils import pcprint, separator
from src.rag import RAG
from src.llm import ClaudeLLM, MistralLLM_BinaryClassification, Llama3LLM_BinaryClassification
from src import eval_utils
from src.eval_utils import ProcessedBenchmarkSample
from src import eval_prompts
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import torch
import re
import time
import sys
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
import shutil
from sklearn.metrics import f1_score


def print_log_summary(logfile):
    import pickle
    from datetime import datetime

    # Load the pickle file
    # file_path = '/mnt/data/mmlu_test_2024-09-12-15-29_logs.pkl'
    file_path = logfile
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Extract necessary fields
    date = datetime.now().strftime('%Y-%m-%d')  # Using current date, or adjust if you have a specific date to use
    llm_type = data['config']['llm']['llm_type']
    model = data['config'][f'llm_{llm_type}']['model_name']
    embedding_model = data['config']['embedding']['model_name']
    dataset = data['dataset']
    split = data['split']
    rag_mode = data['rag_mode']
    top_k = data['top_k']
    score = data['eval_scores']['accuracy']

    # Print in the required format
    print(f"{date},{model},{embedding_model},{dataset},{split},{rag_mode},{top_k},{score:.4f}")



def process_sample(rag, sample, in_context_mode, top_k, answer_col):
    """
    Process a single sample to retrieve relevant extra info from db (rag).
    """
    s = eval_utils.process_medhallu_qa_sample(sample, answer_col=answer_col)
    
    try:
        if in_context_mode:
            rag_examples = rag(s["question"], topk=top_k) # Question-Only Retrieval (QOR): to align with real-world cases of medical QA, answer options should not be provided as input during retrieval.
            # rag_examples_str = eval_utils.process_rag_sample(rag_examples)
            rag_examples_str = eval_utils.process_rag_sample_for_medhallue(rag_examples)

            # SYSTEM_PROMPT = eval_prompts.SYSTEM_PROMPT_RAG
            SYSTEM_PROMPT = eval_prompts.MEDHALLU_SYSTEM_PROMPT
            USER_PROMPT = eval_prompts.MEDHALLU_USER_PROMPT.format(
                rag_examples_str, s["question"], s["answer"] # world_knowledge, question, answer
            )

        else:
            SYSTEM_PROMPT = eval_prompts.MEDHALLU_SYSTEM_PROMPT_WORAG
            USER_PROMPT = eval_prompts.MEDHALLU_USER_PROMPT_WORAG.format(s["question"], s["answer"]) # question, answer, without RAG knowledge
            rag_examples = None
    except Exception as e:
        print(colored(f"Error processing sample during RAG retrieval: {e}", "red"))
        SYSTEM_PROMPT = eval_prompts.MEDHALLU_SYSTEM_PROMPT_WORAG
        USER_PROMPT = eval_prompts.MEDHALLU_USER_PROMPT_WORAG.format(s["question"], s["answer"])
        rag_examples = None
        
    prompts = {"system": SYSTEM_PROMPT, "user": USER_PROMPT}
    
    if answer_col.lower() == 'ground truth':
        gold_label = '0' # not hallucinated
    elif answer_col.lower() == 'hallucinated answer':
        gold_label = '1' # hallucinatd

    return [ProcessedBenchmarkSample(
            sample_id=sample["id"],
            rag_examples=rag_examples,
            prompts=prompts,
            gold_label=gold_label,
        )], [prompts]

def chunk_prompts(prompts, world_size):
    chunk_size = len(prompts) // world_size
    chunks = [prompts[i * chunk_size: (i + 1) * chunk_size] for i in range(world_size)]
    # Add remaining prompts to the last chunk
    if len(prompts) % world_size != 0:
        chunks[-1].extend(prompts[world_size * chunk_size:])
    return chunks

def run_model_initialization_ddp_generation(rank, world_size, config, prompts_chunks):

    # Initialize the distributed process group
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    # get one chunk of prompts
    prompts_chunk = prompts_chunks[rank]
    try:
        # Init the model
        if config["llm"]["llm_type"] == 'mistral':
            model = MistralLLM_BinaryClassification(model_name=config["llm_mistral"]["model_name"],
                                batch_size=config["llm_mistral"]["batch_size"],
                                max_new_tokens=config["llm_mistral"]["max_new_tokens"],
                                temperature=config["llm_mistral"]["temperature"],
                                torch_dtype=torch.bfloat16,
                                load_in_8bit=config["llm_mistral"]["load_in_8bit"],
                                max_length=config["llm_mistral"]["max_length"],
                                rank=rank,
                                world_size=world_size)
        elif config["llm"]["llm_type"] == 'llama3':
            model = Llama3LLM_BinaryClassification(model_name=config["llm_llama3"]["model_name"],
                                batch_size=config["llm_llama3"]["batch_size"],
                                max_new_tokens=config["llm_llama3"]["max_new_tokens"],
                                temperature=config["llm_llama3"]["temperature"],
                                torch_dtype=torch.bfloat16,
                                load_in_8bit=config["llm_llama3"]["load_in_8bit"],
                                max_length=config["llm_llama3"]["max_length"],
                                rank=rank,
                                world_size=world_size)

        # Generate responses for the chunk of prompts
        local_responses = model.generate(prompts_chunk, rank=rank, world_size=world_size)
        # Save local responses to a file
        if not os.path.exists("model_responses"):
            os.makedirs("model_responses")
        with open(f"model_responses/{config['llm']['llm_type']}_responses_rank_{rank}.json", "w") as f:
            json.dump(local_responses, f)
            
    except Exception as e:
            print(f"Error in rank {rank}: {e}")
            
    finally:
        # Finalize process grou  p
        dist.destroy_process_group()
    
def get_preds_and_evals(config, dataset, in_context_mode, top_k):
    embedding_model = config["embedding"]["model_name"] 
    model_name_short = embedding_model.split("/")[-1]
    content = config["embedding"]["content"]
    miriad_version = config["embedding"]["miriad_version"]
    if miriad_version == '4.8M':
        collection_name = config["qdrant"]["collection_4.4M"].format(model_name=model_name_short, content=content)
    else:
        collection_name = config["qdrant"]["collection_5.8M"].format(model_name=model_name_short, content=content)
    
    rag = RAG(qdrant_host=config["qdrant"]["host"],
            qdrant_port=config["qdrant"]["port"],
            db_name=collection_name,
            embedding_model=embedding_model)

    
    print(colored(dataset, "blue"))
    raged_samples = []
    list_of_prompts = []
    # sequential processing of retrieval
    for id, sample in tqdm(enumerate(dataset), desc="Processing rag retrieval for each sample"):
        # raged_sample_entry, prompts = process_sample(rag, sample, in_context_mode, top_k, answer_col) # prev
        for c, answer_col in enumerate(['Ground Truth', 'Hallucinated Answer']):
            sample['id'] = f"{id}_w{answer_col.replace(' ', '')}"
            raged_sample_entry_list, prompts_list = process_sample(rag, sample, in_context_mode, top_k, answer_col)
            raged_samples.extend(raged_sample_entry_list)
            list_of_prompts.extend(prompts_list)
        
    # Backbone LLM processing: async API call for Claude model or parallel processing for other local models (Mistral)
    if config["llm"]["llm_type"] == 'claude':
        language_model = ClaudeLLM(model_name=config["llm_claude"]["model_name"]) # Initialize the Claude model
        responses = language_model.generate(list_of_prompts)
    elif config["llm"]["llm_type"] == 'mistral' or config["llm"]["llm_type"] == 'llama3':
        world_size = len(config['llm']['available_devices']) if config['llm']['available_devices'] is not None else torch.cuda.device_count()
        print(f"Using GPU: {config['llm']['available_devices']}")
        prompts_chunks = chunk_prompts(list_of_prompts, world_size)
        # Launch DDP processes
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        mp.spawn(run_model_initialization_ddp_generation, args=(world_size, config, prompts_chunks), nprocs=world_size)
        # Read and aggregate responses
        responses = []
        for r in range(world_size):
            res_file = f"model_responses/{config['llm']['llm_type']}_responses_rank_{r}.json"
            with open(res_file, "r") as f:
                local_res = json.load(f)
            responses.extend(local_res)
            # # Cleanup the responses file
            # os.remove(res_file)

    assert len(responses) == len(raged_samples)
    for i, res in enumerate(responses):
        raged_samples[i].llm_res = res
        try:
            # raged_samples[i].pred_label = eval_utils.letter2int(res["choice"])
            raged_samples[i].pred_label = res["choice"]
        except Exception as e:
            print(f"Something's wrong with the response: {res}")
            raged_samples[i].pred_label = "-1"
        raged_samples[i].is_correct = raged_samples[i].pred_label == raged_samples[i].gold_label
        
    return raged_samples

def summarize_evaluation(processed_samples):
    """
    Summarize evaluation results from processed_samples.
    """
    correct = sum(1 for sample in processed_samples if sample.is_correct)
    total = len(processed_samples)
    accuracy = correct / total if total > 0 else 0.0
    
    # calculate the F1 score
    y_true = [int(sample.gold_label) for sample in processed_samples]
    y_pred = [int(sample.pred_label) for sample in processed_samples]

    f1 = f1_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    
    print(colored(f"Accuracy: {accuracy * 100:.3f}%", "green"))
    print(colored(f"F1 Score: {f1 * 100:.3f}%", "green"))
    return {"accuracy": accuracy, "f1_score": f1}


def main(eval_dataset, split, in_context_mode, top_k=3):
    """
    Evaluates the dataset by generating predictions for each sample.

    Args:
        eval_dataset (str): Name of the dataset to be evaluated, e.g., 'mmlu'.
        split (str): The specific dataset split to evaluate; options are 'train', 'dev', or 'test'.
        in_context_mode (bool): Whether to use in-context learning (ie RAG augmented) or not. If False, `top_k` is ignored.
        top_k (int): The number of top predictions to consider when processing the dataset.

    Returns:
        A list of predictions for each dataset sample.
    """
    print(f"Geeettt ready!")
    
    # load config.yaml file from the parent directory
    config_path = "./medhallu_eval_config.yaml"
    # load the config file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("Loaded eval config:")
    pcprint(config)
    
    # start a new wandb run to track this script
    os.environ["WANDB_DIR"] = "/tmp"
    embedding_models = config["embedding"]["model_names"]
    backbone_llm_types = config["llm"]["llm_types"]
    rag_contents = config["embedding"]["contents"]
    benchmark_name = eval_dataset
    
    dataset = eval_utils.load_standardize_dataset(eval_dataset, split)
    
    print(colored(f"Finished loading Dataset: {eval_dataset}!!!!!!!!!", "blue"))
    
    for embedding_model in embedding_models:
        for backbone_llm_type in backbone_llm_types:
            num_rag_content_processed = 0
            for rag_mode in rag_contents:
                if not in_context_mode and num_rag_content_processed == 1:
                    break
                if not in_context_mode:
                    rag_mode = "woRAG"
                config["embedding"]["model_name"] = embedding_model
                config["llm"]["llm_type"] = backbone_llm_type
                config["embedding"]["content"] = rag_mode
                
                processed_samples = get_preds_and_evals(config, dataset, in_context_mode, top_k)
                eval_scores = summarize_evaluation(processed_samples)
                print(colored(f"Evaluation: {str(eval_scores)}", "green"))
                # wandb.log({"acc": eval_scores, "in_context_mode": in_context_mode, "num_samples": len(processed_samples)})
                
                # log output
                logs = {
                    "dataset": eval_dataset,
                    "rag_mode": rag_mode,
                    "top_k": top_k if in_context_mode else None,
                    "split": split,
                    "processed_samples": processed_samples, # contains all the preds too
                    "eval_scores": eval_scores,
                    "config": config,
                }
                if not os.path.exists("medhallu_logs"):
                    os.makedirs("medhallu_logs")
                embedding_content = config["embedding"]["content"] if in_context_mode else "woRAG"
                # Save logs to a pickle file
                logfile = f"medhallu_logs/{eval_dataset}_{split}_{config['embedding']['model_name'].split('/')[-1]}_miriad_{config['embedding']['miriad_version']}_{embedding_content}_{config['llm']['llm_type']}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}_logs_1855_top{top_k}.pkl"
                with open(logfile, "wb") as f:
                    pickle.dump(logs, f)

                # print_log_summary(logfile)
                # wandb.finish() # finish the wandb run
                shutil.rmtree('./tmp', ignore_errors=True)
                
                num_rag_content_processed += 1

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    if len(sys.argv) > 1:  # Check if command-line arguments are provided
        fire.Fire(main)
    else:  # Fall back to manual invocation
        print('oh hey here we go to the main function')
        main(eval_dataset="medhallu", split="full", in_context_mode=True, top_k=20)