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
from src.llm import ClaudeLLM, MistralLLM, Llama3LLM
from src import eval_utils
from src.eval_utils import ProcessedBenchmarkSample
from src import eval_prompts
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
import json
import torch
import re
import time
import sys
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
import shutil
import tiktoken
import glob
from pathlib import Path

def print_log_summary(logfile):
    import pickle
    from datetime import datetime

    # Load the pickle file
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

def check_existing_processed_combinations(log_dir):
    log_pattern = re.compile(
        r"(?P<eval_dataset>.+?)_"                     # dataset
        r"(?P<split>.+?)_"                            # split
        r"(?P<embedding_model>[^_]+)_"                # embedding model (assumes no underscores)
        r"miriad_5\.8M_"                              
        r"(?P<embedding_content>.+?)_"                # embedding content (can have underscores)
        r"(?P<llm_type>[^_]+)_"                       # llm type (assumes no underscores)
        r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}_"             # timestamp
        r"logs_1855_"
        r"(?P<n_tokens>\d+)tokens\.pkl"
    )

    processed_combinations = set()

    for log_file in Path(log_dir).glob("*.pkl"):
        match = log_pattern.match(log_file.name)
        if not match:
            print(f"Skipping unmatched filename: {log_file.name}")
            continue
        
        key = (
            match.group("eval_dataset"),
            match.group("split"),
            match.group("embedding_model"),
            match.group("embedding_content"),
            match.group("llm_type"),
            match.group("n_tokens")
        )
        processed_combinations.add(key)
    
    return processed_combinations



def process_sample(rag, sample, in_context_mode, top_k, passage_text_comparison_mode, tokenizer, n_tokens, answer_col="answer"):
    """
    Process a single sample to retrieve relevant extra info from db (rag).
    """
    s = eval_utils.process_mcq_sample(sample)
    
    try:
        if in_context_mode and not passage_text_comparison_mode:
            rag_examples = rag(s["question"], topk=top_k) # Question-Only Retrieval (QOR): to align with real-world cases of medical QA, answer options should not be provided as input during retrieval.
            rag_examples_str, actual_rag_examples = eval_utils.process_rag_sample(rag_examples) # just to avoid redundancy in rag_examples
            rag_examples = actual_rag_examples

            # SYSTEM_PROMPT = eval_prompts.SYSTEM_PROMPT_RAG
            SYSTEM_PROMPT = eval_prompts.SYSTEM_PROMPT_RAG_MISTRAL
            USER_PROMPT = eval_prompts.USER_PROMPT_RAG.format(
                rag_examples_str, s["question"], s["choices"]
            )
        elif in_context_mode and passage_text_comparison_mode:
            if passage_text_comparison_mode == 'k-passages':
                rag_examples = rag(s["question"], topk=top_k*3) # topk*3 to make sure we have enough passage sources
                rag_examples_str, rag_examples = eval_utils.process_rag_sample_k_passages(rag_examples, k_passages=top_k) # processed_rag_examples_str, actually used rag_examples
            elif passage_text_comparison_mode == 'n-tokens':
                rag_examples = rag(s["question"], topk=20) # topk = 20 to make sure we have enough passage sources
                rag_examples_str, rag_examples = eval_utils.process_rag_sample_n_tokens(rag_examples, tokenizer, n_tokens=n_tokens)
                
            # SYSTEM_PROMPT = eval_prompts.SYSTEM_PROMPT_RAG
            SYSTEM_PROMPT = eval_prompts.SYSTEM_PROMPT_RAG_MISTRAL
            USER_PROMPT = eval_prompts.USER_PROMPT_RAG.format(
                rag_examples_str, s["question"], s["choices"]
            )
        else:
            # print(colored("Running wo RAG", "yellow"))
            SYSTEM_PROMPT = eval_prompts.SYSTEM_PROMPT
            USER_PROMPT = eval_prompts.USER_PROMPT.format(s["question"], s["choices"])
            rag_examples = None
    except Exception as e:
        print(colored(f"Error processing sample during RAG retrieval: {e}", "red"))
        SYSTEM_PROMPT = eval_prompts.SYSTEM_PROMPT
        USER_PROMPT = eval_prompts.USER_PROMPT.format(s["question"], s["choices"])
        rag_examples = None
        
    prompts = {"system": SYSTEM_PROMPT, "user": USER_PROMPT}


    return ProcessedBenchmarkSample(
            sample_id=sample["id"],
            rag_examples=rag_examples,
            prompts=prompts,
            gold_label=sample[answer_col],
        ), prompts


def chunk_prompts(prompts, world_size):
    chunk_size = len(prompts) // world_size
    chunks = [prompts[i * chunk_size: (i + 1) * chunk_size] for i in range(world_size)]
    # Add remaining prompts to the last chunk
    if len(prompts) % world_size != 0:
        chunks[-1].extend(prompts[world_size * chunk_size:])
    return chunks

def run_model_initialization_ddp_generation(rank, world_size, config, prompts_chunks):
    device_id = rank  # We use the rank directly because CUDA_VISIBLE_DEVICES has remapped devices
    
    # Set the device - this should now be a valid ordinal (0, 1, 2, etc.)
    torch.cuda.set_device(device_id)
    # Log which physical GPU we're actually using
    physical_id = config['llm']['available_devices'][rank] if config['llm']['available_devices'] is not None else rank
    print(f"Process {rank} using GPU {physical_id} (device ordinal {device_id})")

    # Initialize the distributed process group
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    # get one chunk of prompts
    prompts_chunk = prompts_chunks[rank]
    try:
        # Init the model
        if config["llm"]["llm_type"] == 'mistral':
            model = MistralLLM(model_name=config["llm_mistral"]["model_name"],
                                batch_size=config["llm_mistral"]["batch_size"],
                                max_new_tokens=config["llm_mistral"]["max_new_tokens"],
                                temperature=config["llm_mistral"]["temperature"],
                                torch_dtype=torch.bfloat16,
                                load_in_8bit=config["llm_mistral"]["load_in_8bit"],
                                max_length=config["llm_mistral"]["max_length"],
                                rank=rank,
                                world_size=world_size,
                                device_id=device_id)
        elif config["llm"]["llm_type"] == 'llama3':
            model = Llama3LLM(model_name=config["llm_llama3"]["model_name"],
                                batch_size=config["llm_llama3"]["batch_size"],
                                max_new_tokens=config["llm_llama3"]["max_new_tokens"],
                                temperature=config["llm_llama3"]["temperature"],
                                torch_dtype=torch.bfloat16,
                                load_in_8bit=config["llm_llama3"]["load_in_8bit"],
                                max_length=config["llm_llama3"]["max_length"],
                                rank=rank,
                                world_size=world_size,
                                device_id=device_id)

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
    
def get_preds_and_evals(config, dataset, in_context_mode, top_k, passage_text_comparison_mode, n_tokens, answer_col="answer"):
    embedding_model = config["embedding"]["model_name"] 
    model_name_short = embedding_model.split("/")[-1]
    content = config["embedding"]["content"]
    if content != 'passage_text' and passage_text_comparison_mode != 'n-tokens':
        passage_text_comparison_mode = None # passage_text_comparison_mode should only be used when content is 'passage_text' for  'k-passages' or when 'n-tokens' for whatever rag content.
    collection_name = config["qdrant"]["collection"].format(model_name=model_name_short, content=content)
    
    rag = RAG(qdrant_host=config["qdrant"]["host"],
            qdrant_port=config["qdrant"]["port"],
            db_name=collection_name,
            embedding_model=embedding_model)

    if passage_text_comparison_mode == 'n-tokens':
        if config["llm"]["llm_type"] == 'mistral':
            tokenizer = AutoTokenizer.from_pretrained(config["llm_mistral"]["model_name"])
        elif config["llm"]["llm_type"] == 'llama3':
            tokenizer = AutoTokenizer.from_pretrained(config["llm_llama3"]["model_name"])
        elif config["llm"]["llm_type"] == 'claude':
            tokenizer = tiktoken.get_encoding("cl100k_base") # closest approximation to closed source claude tokenizer
    else:
        tokenizer = None
        
    
    print(colored(dataset, "blue"))
    raged_samples = []
    list_of_prompts = []
    # sequential processing of retrieval
    for sample in tqdm(dataset, desc="Processing rag retrieval for each sample"):
        raged_sample_entry, prompts = process_sample(rag, sample, in_context_mode, top_k, passage_text_comparison_mode, tokenizer, n_tokens, answer_col)
        raged_samples.append(raged_sample_entry)
        list_of_prompts.append(prompts)
        
    # Backbone LLM processing: async API call for Claude model or parallel processing for other local models (Mistral)
    if config["llm"]["llm_type"] == 'claude':
        print(colored("Using Claude model for generation", "green"))
        language_model = ClaudeLLM(model_name=config["llm_claude"]["model_name"]) # Initialize the Claude model
        responses = language_model.generate(list_of_prompts)
        print(colored(f"Claude model generation completed, got {len(responses)} responses", "green"))
    elif config["llm"]["llm_type"] == 'mistral' or config["llm"]["llm_type"] == 'llama3':
        world_size = len(config['llm']['available_devices']) if config['llm']['available_devices'] is not None else torch.cuda.device_count()
        print(f"Using GPUs: {config['llm']['available_devices'] if config['llm']['available_devices'] is not None else list(range(world_size))}")
        prompts_chunks = chunk_prompts(list_of_prompts, world_size)
        # Launch DDP processes
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        # Set visible devices if specified
        if config['llm']['available_devices'] is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config['llm']['available_devices']))
            physical_gpu_ids = list(range(len(config['llm']['available_devices'])))
        else:
            print(f"Using all available GPUs: {list(range(torch.cuda.device_count()))}")
            physical_gpu_ids = None
        mp.spawn(run_model_initialization_ddp_generation, args=(world_size, config, prompts_chunks, physical_gpu_ids), nprocs=world_size)
        # Read and aggregate responses
        responses = []
        for r in range(world_size):
            res_file = f"model_responses/{config['llm']['llm_type']}_responses_rank_{r}.json"
            with open(res_file, "r") as f:
                local_res = json.load(f)
            responses.extend(local_res)
            # # Cleanup the responses file
            # os.remove(res_file)
    
    print(f"len responses: {len(responses)}")
    print(f"len raged_samples: {len(raged_samples)}")
    assert len(responses) == len(raged_samples)
    for i, res in enumerate(responses):
        raged_samples[i].llm_res = res
        try:
            raged_samples[i].pred_label = eval_utils.letter2int(res["choice"])
        except Exception as e:
            print(f"Something's wrong with the response: {res}")
            raged_samples[i].pred_label = eval_utils.letter2int(None)
        raged_samples[i].is_correct = raged_samples[i].pred_label == raged_samples[i].gold_label
        
    return raged_samples

def summarize_evaluation(processed_samples):
    """
    Summarize evaluation results from processed_samples.
    """
    correct = sum(1 for sample in processed_samples if sample.is_correct)
    total = len(processed_samples)
    accuracy = correct / total if total > 0 else 0.0
    print(colored(f"Accuracy: {accuracy * 100:.3f}%", "green"))
    return {"accuracy": accuracy}


def main(eval_dataset, split, in_context_mode=True, top_k=3, passage_text_comparison_mode='n-tokens', n_tokens=1000):
    """
    Evaluates the dataset by generating predictions for each sample.

    Args:
        eval_dataset (str): Name of the dataset to be evaluated, e.g., 'mmlu'.
        split (str): The specific dataset split to evaluate; options are 'train', 'dev', or 'test'.
        in_context_mode (bool): Whether to use in-context learning (ie RAG augmented) or not. If False, `top_k` is ignored.
        top_k (int): The number of top predictions to consider when processing the dataset.
        passage_text_comparison_mode (str): The mode for comparing passage text; options are 'k-passages' or 'n-tokens'. Only used when in_context_mode is True and rag content is 'passage_text'.
        n_tokens (int): The number of tokens to consider when processing the rag content. Only used when in_context_mode is True and passage_text_comparison_mode is 'n-tokens'.

    Returns:
        A list of predictions for each dataset sample.
    """
    
    # load config.yaml file from the parent directory
    config_path = "./eval_config.yaml"
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
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project=project_name,

    #     # track hyperparameters and run metadata
    #     config={
    #         "embedding_model": config["embedding"]["model_name"],
    #         "backbone_llm_type": config["llm"]["llm_type"], 
    #         "benchmark_dataset": eval_dataset,
    #         "split": split,
    #     },
    #     name=f"{rag_mode}_miriad_4.4M_top{top_k}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}",
    # )

    dataset = eval_utils.load_standardize_dataset(eval_dataset, split)
    
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
                
                # search for the settings that've been done in the past (judging by the logs) and skip to avoid reprocessing
                log_dir = "token_parity_logs"
                processed_combinations = check_existing_processed_combinations(log_dir)
                current_combo = (eval_dataset, split, config['embedding']['model_name'].split('/')[-1], rag_mode, config['llm']['llm_type'], str(n_tokens))
                if current_combo in processed_combinations:
                    print(f"Skipping already processed combination: {current_combo}")
                    continue
                # print(colored(f"Processed combinations\n {processed_combinations}", "blue"))
                print(colored(f"Processing combination: {current_combo}", "blue"))
                processed_samples = get_preds_and_evals(config, dataset, in_context_mode, top_k, passage_text_comparison_mode, n_tokens=n_tokens)
                eval_scores = summarize_evaluation(processed_samples)
                print(colored(f"Evaluation: {str(eval_scores)}", "green"))
                # wandb.log({"acc": eval_scores, "in_context_mode": in_context_mode, "num_samples": len(processed_samples)})
                # log output
                logs = {
                    "dataset": eval_dataset,
                    "rag_mode": in_context_mode,
                    "top_k": top_k if in_context_mode else None,
                    "split": split,
                    "processed_samples": processed_samples, # contains all the preds too
                    "eval_scores": eval_scores,
                    "config": config,
                }
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                embedding_content = config["embedding"]["content"] if in_context_mode else "woRAG"
                # Save logs to a pickle file
                logfile = f"{log_dir}/{eval_dataset}_{split}_{config['embedding']['model_name'].split('/')[-1]}_miriad_5.8M_{embedding_content}_{config['llm']['llm_type']}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}_logs_1855_{n_tokens}tokens.pkl"
                with open(logfile, "wb") as f:
                    pickle.dump(logs, f)

                # print_log_summary(logfile)
                # wandb.finish() # finish the wandb run
                shutil.rmtree('./tmp', ignore_errors=True)
                
                num_rag_content_processed += 1

if __name__ == "__main__":
    if len(sys.argv) > 1:  # Check if command-line arguments are provided
        fire.Fire(main)
    else:  # Fall back to manual invocation
        top_k = 20
        n_tokens = [200, 600, 1000, 1400, 1800, 2200, 2600]
        for n in n_tokens:
            main(eval_dataset='medmcqa', split="dev", in_context_mode=False, top_k=top_k, passage_text_comparison_mode='n-tokens', n_tokens=n)