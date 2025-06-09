import json
import os
import logging
from tqdm import tqdm
import tiktoken 
import re
import sys

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

tokenizer = tiktoken.get_encoding("cl100k_base")

# Function to load a JSON file
def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)

# Function to split text into smaller chunks based on a maximum number of tokens
def split_text(text, tokenizer, max_tokens):
    # Split the text into sentences using regular expressions
    sentences = re.split('(?<=[.!?]) +', text)
    # Count the number of tokens in each sentence
    n_tokens = [len(tokenizer.encode(sentence)) for sentence in sentences]

    chunks = []  # List to hold chunks of text
    tokens_so_far = 0  # Counter for tokens in the current chunk
    chunk = []  # List to hold sentences for the current chunk

    # Loop through each sentence and its token count
    for sentence, token in zip(sentences, n_tokens):
        
        # Check if adding the current sentence would exceed the max tokens for the chunk
        if tokens_so_far + token > max_tokens:
            chunks.append(" ".join(chunk).strip())
            chunk = []
            tokens_so_far = 0

        # Skip sentences that themselves exceed the max token count
        if token > max_tokens:
            continue

        # Add the current sentence to the chunk
        chunk.append(sentence)
        tokens_so_far += token

    # Add any remaining sentences as a final chunk
    if chunk:
        chunks.append(" ".join(chunk).strip())

    return chunks


if __name__ == "__main__":
    # Get start and end shard numbers from command-line arguments
    start_shard = int(sys.argv[1])
    end_shard = int(sys.argv[2])

    # Loop through specified range of shards
    for i in tqdm(range(start_shard, end_shard + 1), desc='Creating Passages'):
        # Skip the 20th shard
        if i == 20:
            continue

        # Load papers from a JSON file
        papers = load_json(f'medicine_papers/medicine_shard_{i}.json')

        passages_info = []  # List to hold processed passages and their metadata

        # Loop through each paper to split its text into chunks
        for paper in papers:
            # Create chunks of text from the paper
            passages = split_text(text=paper['paper'], tokenizer=tokenizer, max_tokens=1000)

            # Create a dictionary with the original text and its metadata
            paper_info = {'text': paper['paper'], 'metadata': paper['metadata']}
            
            # Append the processed paper and its passages to the list
            passages_info.append({'paper': paper_info, 'passages': passages})

        # Save the processed passages to a new JSON file
        with open(f'medicine_passages/medicine_passages_{i}.json', 'w') as f:
            json.dump(passages_info, f)