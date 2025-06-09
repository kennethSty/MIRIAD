from datasets import load_dataset, Dataset
from tqdm import tqdm
import json
import os
import re

directory = "relevance_results"
good_qa_ids = []

for filename in tqdm(os.listdir(directory)):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path) and filename.endswith(".json"):  # Ensure it's a JSON file
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for qa_id, text in data.items():
                if re.search(r"Category:\s*good", text, re.IGNORECASE):  # Case-insensitive match
                    good_qa_ids.append(qa_id)

# Print all extracted QA IDs with "Good" category
print(len(good_qa_ids))
print(len(set(good_qa_ids)))

good_qa_ids = set(good_qa_ids)

dataset = load_dataset("miriad/miriad-v0.1.1-5.8M-with-chunks")['train']

filtered_dataset = dataset.filter(
    lambda example: example['qa_id'] in good_qa_ids, 
    num_proc=128  # Adjust this number based on your CPU cores
)

# Push the filtered dataset to the Hugging Face Hub
filtered_dataset.push_to_hub("miriad/miriad-v0.2.1-4.4M-with-chunks")