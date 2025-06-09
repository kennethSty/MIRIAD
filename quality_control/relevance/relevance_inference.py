import os
import argparse
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset

def set_environment(gpu_id):
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'

def split_dataset(dataset, num_splits):
    split_size = len(dataset) // num_splits
    splits = [dataset.select(range(i * split_size, (i + 1) * split_size)) for i in range(num_splits - 1)]
    splits.append(dataset.select(range((num_splits - 1) * split_size, len(dataset))))
    return splits

def format_prompt(question, answer):
    task = f'''
    I took passages of text from the medical literature and converted each passage into Q&A problems with the goal to distill the medical knowledge
    of the literature into a set of Q&A problems. Having generated them, I noticed that some Q&As are bad because they do not contain medically or
    biomedically relevant information. 
    Ideally, a Q&A conveys a relevant piece of medical knowledge like this good example: 
    
    "Q: What is the first-line therapy for M. tuberculosis pneumonia? 
    A: Rifampin, isoniazid, pyrazinamide, and ethambutol are first-line antitubercular medications."
    
    By contrast, a bad Q&A would have one or more of the following issues:
    1. Refers to the details of a specific study (that is presented in the passage), such as specific details about the study's experimental design,
    the used statistical methods used in this study, tables or figures that appear in it, study dates, locations, funding sources, or other details
    that are not essential for understanding the underlying medical facts.
    2. Is heavily dependent on study-specific details that cannot be understood without the original passage, such as discussing the study's
    specific findings, limitations, or conclusions without providing sufficient background information.
    3. Focuses on experimental methods or protocols that, while medically related, are too specific to the referenced study and do not convey
    broadly relevant medical knowledge. 
    
    If the Q&A does not have any other bad aspects mentioned above, then the Q&A should be classified as good. A Q&A that effectively communicates a
    clinical procedure, treatment approach, or any clinical or biomedical knowledge in a clear and concise manner should be considered good.
    
    Use the above criteria to judge whether the following Q&A is good or bad. Classify the Q&A as either "good" or "bad" and provide a short
    explanation for the classification.
    
    Explanation: {{explanation}}
    Category: {{good or bad}}

    Question: {question}
    Answer: {answer}
    '''

    prompt = f'''
    <s>[INST] { task } [/INST]
    '''
    return prompt

def prepare_prompts(dataset_split):
    prompts_all = []
    qa_ids = []

    for sample in dataset_split:
        qa_ids.append(sample['qa_id'])
        question = sample['question']
        answer = sample['answer']
        prompt = format_prompt(question, answer)
        prompts_all.append(prompt)
    
    return prompts_all, qa_ids

def prepare_prompts_tokenized(prompts, tokenizer, batch_size=64):
    batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    batches_tok = []
    tokenizer.padding_side = "left"
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding='longest',
                truncation=False,
                pad_to_multiple_of=8,
                add_special_tokens=False
            ).to("cuda")
        )
    tokenizer.padding_side = "right"
    return batches_tok

def run_model_on_prompts(model, tokenizer, prompts, batch_size, output_file=None, qa_ids=None):
    results = {}
    prompt_batches = prepare_prompts_tokenized(prompts, tokenizer, batch_size=batch_size)
    processed_qa_ids = set()

    # Check if partial results exist
    if output_file and os.path.exists(output_file):
        with open(output_file, 'r') as f:
            partial_results = json.load(f)
            results = {qa_id: response for qa_id, response in partial_results.items()}
            processed_qa_ids = set(results.keys())

    # Calculate the start index based on processed QA IDs
    start_batch_idx = len(processed_qa_ids) // batch_size

    for i in tqdm(range(start_batch_idx, len(prompt_batches))):
        prompts_tokenized = prompt_batches[i]
        outputs_tokenized = model.generate(**prompts_tokenized, max_new_tokens=200)
        outputs_tokenized = [tok_out[len(tok_in):] for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized)]
        outputs = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
        batch_results = {qa_ids[i * batch_size + j]: output for j, output in enumerate(outputs)}
        results.update(batch_results)
        # Move prompts_tokenized to CPU to free GPU memory
        # Save intermediate results
        if output_file and (i + 1) % 10 == 0:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)

    # Final save
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', required=True, type=int, help="Dataset split part number")
    parser.add_argument('--gpu', required=True, type=int, help="GPU ID")
    args = parser.parse_args()

    set_environment(args.gpu)

    # Load dataset
    dataset = load_dataset("miriad/miriad-v0.1-5.8M")['train']
    dataset_splits = split_dataset(dataset, 16)  # Split into 16 parts
    args.part += 8
    dataset_part = dataset_splits[args.part]  # Select the part for this process

    # Prepare prompts and qa_ids
    prompts_all, qa_ids = prepare_prompts(dataset_part)

    # Load model and tokenizer
    model_path = "finetuned_models/relevance_batch8"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    output_file = f"relevance_results/split_{args.part}.json"
    
    # Run model on prompts
    results = run_model_on_prompts(model, tokenizer, prompts_all, batch_size=64, output_file=output_file, qa_ids=qa_ids)

    # Save results with qa_ids
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
