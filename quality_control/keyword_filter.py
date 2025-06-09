import json
from tqdm import tqdm
import os
from collections import defaultdict
from datasets import load_dataset, Dataset
import re

passage_reference_phrases = [r"the passage", r"the study"]

directory_path = '../data_generation/generated_data'

passage_reference_pattern = re.compile("|".join(passage_reference_phrases), re.IGNORECASE)

filtered_dataset = []

for filename in tqdm(os.listdir(directory_path), desc="loading files"):

    filepath = os.path.join(directory_path, filename)

    if '33' in filepath:
        continue

    with open(filepath, 'r') as f:
        dataset = json.load(f)

    for paper_id in dataset:
        paper = dataset[paper_id]
        year = paper['paper_info']['metadata']['year']

        for passage_id in paper['passage_info']:
            qa = paper['passage_info'][passage_id]['qa']
            passage_text = paper['passage_info'][passage_id]['passage_text']
            metadata = paper['passage_info'][passage_id]['metadata']
            passage_position = passage_id.split('_')[-1].strip(': ')
            
            for i, (qa_id, pair) in enumerate(qa.items()):
                q = pair['question']
                a = pair['answer']
                if passage_reference_pattern.search(pair['answer']):
                    continue


                current = {
                    "qa_id": f"{passage_id}_{qa_id}",
                    "paper_id": paper_id,
                    "question": q,
                    "answer": a,
                    "paper_url": metadata['paper_url'],
                    "paper_title": metadata['paper_title'],
                    "passage_text": passage_text,
                    "passage_position": passage_position,
                    "year": year

                }

                filtered_dataset.append(current)



data_dict = {key: [dic[key] for dic in filtered_dataset] for key in filtered_dataset[0]}

dataset = Dataset.from_dict(data_dict)

dataset.push_to_hub("miriad/miriad-v0.1-5.8M")

