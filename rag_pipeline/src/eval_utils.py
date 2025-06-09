from datasets import load_dataset, concatenate_datasets
from typing import List, Dict, Optional
from dataclasses import dataclass
from transformers import PreTrainedTokenizerFast
import tiktoken
from termcolor import colored

@dataclass
class ProcessedBenchmarkSample:
    """Standardized format for processed benchmark sample."""
    sample_id: str
    rag_examples: List
    prompts: Dict
    gold_label: int
    llm_res: Optional[Dict] = None
    pred_label: Optional[int] = None
    is_correct: Optional[bool] = None


def int2letter(n):
    return chr(65 + n)


def letter2int(value):
    # Check if the input is a single character and a letter
    if isinstance(value, str):
        if value.isalpha() and len(value) == 1:
            return ord(value.upper()) - 65
        else:
            return letter2int("Z") # If it's "none" or something else, return Z
    elif isinstance(value, int):
        return value
    elif value is None:
        return letter2int("Z")
    else:
        raise ValueError(f"Invalid input: {value}")


def process_mcq_sample(sample):
    question = sample["question"]
    choices = "\n".join(
        [f"{int2letter(i)}. {c}" for i, c in enumerate(sample["choices"])]
    )
    as_string = f"# Question:\n{question}\n\n# Choices:\n{choices}".strip()
    return {"question": question, "choices": choices, "as_string": as_string}

def process_medhallu_qa_sample(sample, answer_col):
    question = sample["Question"]
    answer = sample[answer_col]
    return {"question": question, "answer": answer}

def process_rag_sample(sample):
    actually_used_rag_examples = []
    if isinstance(sample, dict):
        sample = [sample]
    out = ""
    prev_passage_text = ""
    num_examples = 0
    for i, s in enumerate(sample):
        current_passage = s.payload['passage_text'].strip()
        if current_passage != prev_passage_text:
            out += f"## Example {i+1}:\n{current_passage}\n\n"
            prev_passage_text = current_passage
            num_examples += 1
            actually_used_rag_examples.append(s)
    return out.strip(), actually_used_rag_examples

def process_rag_sample_n_tokens(sample, tokenizer, n_tokens):
    if isinstance(sample, dict):
        sample = [sample]
    out = ""
    prev_passage_text = ""
    non_redundant_rag_samples = []
    non_redundant_sample_text = []
    for i, s in enumerate(sample):
        current_passage = s.payload['passage_text'].strip()
        if current_passage != prev_passage_text:
            prev_passage_text = current_passage
            non_redundant_rag_samples.append(s) # prepare for rag examples log
            non_redundant_sample_text.append(current_passage) # prepare for tokenization
    if isinstance(tokenizer, tiktoken.Encoding):
        tokenized = tokenizer.encode_batch(non_redundant_sample_text)
        token_ids_list = tokenized
    elif isinstance(tokenizer, PreTrainedTokenizerFast):
        tokenized = tokenizer(non_redundant_sample_text, padding=False, truncation=False, return_attention_mask=False) # tokenized, non_redundant_sample_text, non_redundant_rag_samples should be of the same length
        token_ids_list = tokenized["input_ids"]
    total_tokens = 0
    num_actual_examples = 0
    actually_used_rag_examples = []
    for j, ids in enumerate(token_ids_list):
        total_tokens += len(ids)
        if total_tokens <= n_tokens:
            out += f"## Example {num_actual_examples+1}:\n{non_redundant_sample_text[j]}\n\n"
            num_actual_examples += 1
            actually_used_rag_examples.append(non_redundant_rag_samples[j])
        else:
            break        
    return out.strip(), actually_used_rag_examples

def process_rag_sample_for_medhallue(sample):
    if isinstance(sample, dict):
        sample = [sample]
    out = ""
    prev_passage_text = ""
    num_examples = 0
    for i, s in enumerate(sample):
        if s.payload['passage_text'].strip() != prev_passage_text:
            out += f"## Knowledge {num_examples+1}:\n{s.payload['passage_text'].strip()}\n\n"
            prev_passage_text = s.payload['passage_text']
            num_examples += 1
    return out.strip()

def get_list_of_rag_content(sample):
    """return list of rag content strings"""
    if isinstance(sample, dict):
        sample = [sample]
    out = []
    prev_passage_text = ""
    num_examples = 0
    for i, s in enumerate(sample):
        if s.payload['passage_text'].strip() != prev_passage_text:
            out.append(f"## Example:\n{s.payload['passage_text'].strip()}\n\n")
            prev_passage_text = s.payload['passage_text']
            num_examples += 1
        if num_examples >= 3:
            break
    return out


def load_standardize_dataset(dataset_name, split):
    """
    Standardize datasets. Important keys:
    - question:str
    - choices:List[str]
    - answer:int

    split will be 'test', 'train', 'dev' ('validation' will point to 'dev')
    """
    
    if "first1k" in split:
        split, firstx = split.split("first1k")[0], "first1k"
    else:
        firstx = False

    from datasets import load_dataset

    if dataset_name.lower() == "mmlu":
        sets = ['anatomy', 'clinical_knowledge', 'college_biology', 'college_medicine', 'medical_genetics', 'professional_medicine']
        assert split == 'test', "We should only use the test split for MMLU"
        all_datasets = []
        for subset in sets:
            ds = load_dataset("cais/mmlu", subset)[split]
            def add_custom_id(example, idx):
                example["id"] = f"{subset}_{idx+1}"
                return example
            ds = ds.map(add_custom_id, with_indices=True)
            all_datasets.append(ds)
        dssplit = concatenate_datasets(all_datasets)
        print(colored(f"Loaded MMLU dataset with {len(dssplit)} samples", 'green'))

    elif dataset_name.lower() == "medmcqa":
        ds = load_dataset("openlifescienceai/medmcqa")
        if split == "dev":
            split = "validation"
        dssplit = ds[split]

        # standardize 'choices' and 'answer' columns
        choices = [[x["opa"], x["opb"], x["opc"], x["opd"]] for x in dssplit]

        dssplit = dssplit.add_column("answer", dssplit["cop"])
        dssplit = dssplit.add_column("choices", choices)
    
    elif dataset_name.lower() == "medqa":
        assert split == 'test', "We should only use the test split for MedQA"
        dssplit = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
        def add_custom_id(example, idx):
            example["id"] = f"{split}_{idx+1}"
            return example
        dssplit = dssplit.map(add_custom_id, with_indices=True)
        # standardize 'choices' and 'answer' columns
        choices = [[x['options']['A'], x['options']['B'], x['options']['C'], x['options']['D']] for x in dssplit]
        dssplit = dssplit.remove_columns("answer")
        answer = [ord(x['answer_idx']) - ord('A') for x in dssplit]
        dssplit = dssplit.add_column("answer", answer)
        dssplit = dssplit.add_column("choices", choices)
        print(colored(f"Loaded MedQA dataset with {len(dssplit)} samples", 'green'))
    
    elif dataset_name.lower() == "medhallu":
        print('start loading dataset UTAustin-AIHealth/MedHallu should be cool right?...')
        if split == "labeled":
            dssplit = load_dataset("UTAustin-AIHealth/MedHallu", 'pqa_labeled', split='train')
        elif split == "artificial":
            dssplit = load_dataset("UTAustin-AIHealth/MedHallu", 'pqa_artificial', split='train')
        elif split == "full":
            ds1 = load_dataset("UTAustin-AIHealth/MedHallu", 'pqa_labeled', split='train')
            ds2 = load_dataset("UTAustin-AIHealth/MedHallu", 'pqa_artificial', split='train')
            dssplit = concatenate_datasets([ds1, ds2])
        print('finish loading dataset UTAustin-AIHealth/MedHallu, move to next step')

    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    # If 'firstx' is not False, return the first 1000 samples
    if firstx:
        dssplit = dssplit.select(range(min(1000, len(dssplit))))

    return dssplit