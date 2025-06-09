# Import necessary libraries
import json
import pandas as pd
import os
import logging
from tqdm import tqdm
import pickle
import sys

# Set up logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

# Define paths and shard count
metadata_path = '20200705v1/selected/metadata/'
pdf_path = '20200705v1/selected/processed_pdf_parses/'

# Function to load JSON file
def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)

# Function to load JSONL file
def load_jsonl(file):
    processed_data = []
    with open(file, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                parsed_object = json.loads(line)
                processed_data.append(parsed_object)
    return processed_data

# Accept start and end shard index from command line
start_shard = int(sys.argv[1])
end_shard = int(sys.argv[2])

# Log the range of shards being processed
logging.info(f'Processing from shards {start_shard} to {end_shard}')

# Loop through each shard to process
for i in tqdm(range(start_shard, end_shard + 1), desc='Processing Shards'):
    # Skip shard 20 as it is corrupted
    if i == 20:
        continue

    # Load data for each shard
    subjects = load_json(metadata_path + f'info_{i}.json')
    shard = load_jsonl(pdf_path + f'processed_text_batch_{i}.json')
    metadatas = load_jsonl(metadata_path + f'metadata_{i}.jsonl.gz')
    
    # Create DataFrame to map fields to paper IDs
    field_to_paper_id_df = pd.DataFrame([{'Field': field, 'Paper ID': paper_ids} for field, paper_ids in subjects['field_to_paper_id'].items()])
    # Extract medicine paper IDs
    medicine_paper_ids = list(field_to_paper_id_df[field_to_paper_id_df['Field'] == 'Medicine']['Paper ID'])[0]
    
    # Filter out non-medicine papers
    medicine_papers = []
    for paper, metadata in zip(shard, metadatas):
        if paper['paper_id'] not in medicine_paper_ids:
            continue
        medicine_papers.append({'paper': paper['text'], 'metadata': metadata})

    # Save medicine papers to a JSON file
    with open(f'medicine_papers/medicine_shard_{i}.json', 'w') as f:
        json.dump(medicine_papers, f)