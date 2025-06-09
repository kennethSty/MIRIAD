import os
import torch
import numpy as np
import argparse
import yaml
from typing import Dict, Any, List

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset, load_from_disk
from tqdm import tqdm

from src.database import QdrantHandlerPassageText

class DistributedEmbedder:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize distributed embedding pipeline
        
        Args:
            config (Dict): Configuration dictionary with embedding parameters
        """
        self.config = config
        self.dataset_name = config.get('dataset_name', 'miriad/miriad-v0.1.1-5.8M-with-chunks')
        self.world_size = config.get('world_size', 8)
        self.model_name = config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.batch_size = config.get('batch_size', 72)
        self.output_dir = config.get('output_dir', 'embedding')
        self.tokenize_max_length = config.get('tokenization', {}).get('max_length', 512)
        self.tokenize_stride = config.get('tokenization', {}).get('stride', 64)
        self.force_retokenize = config.get('force_retokenize', False)
        self.content = config.get('content', 'passage_text') # 'passage_text' for passage_text, 'question' for question, 'answer' for answer, 'qa' for question + answer
        self.tokenized_dataset_path = config.get('tokenized_dataset_path', \
                                                'tokenized_dataset/tokenized_miriad_{model_name}_{content}_{max_length}_{stride}').format(
                                                    model_name = self.model_name.split('/')[-1],
                                                    content = self.content,
                                                    max_length=self.tokenize_max_length, 
                                                    stride=self.tokenize_stride
                                                )
        
        # Create output directory if not exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Tokenizer and pre-tokenization setup
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def tokenize_dataset(self):
        """
        Tokenize dataset before distributed processing
        
        Args:
            dataset (Dataset): Input dataset
            max_length (int): Maximum sequence length
        
        Returns:
            Dataset: Tokenized dataset
        """
        def sliding_window_tokenization(batch):
            if self.content != 'qa':
                tokenization_content = batch[self.content]
            elif self.content == 'qa':
                tokenization_content = [f"Question: {q}\nAnswer: {a}" for q, a in zip(batch['question'], batch['answer'])]
            tokenized = self.tokenizer(
                tokenization_content,  # List of long passages in the batch
                truncation=True,
                padding="max_length",
                max_length=self.tokenize_max_length,
                stride=self.tokenize_stride,  # Overlap for sliding window
                return_overflowing_tokens=True,  # Handle overflow chunks
                return_offsets_mapping=False  # Optional, for tracking positions
            )
            
            overflow_to_sample_mapping = tokenized["overflow_to_sample_mapping"]
            # Initialize a counter for each passage
            segment_counters = {}

            # Create unique segment IDs
            segment_ids = []
            batch_qa_id = batch['qa_id']
            for idx in overflow_to_sample_mapping:
                # Initialize the counter for the passage if not already
                if idx not in segment_counters:
                    segment_counters[idx] = 0
                # Generate the segment ID
                segment_id = f"{batch_qa_id[idx]}-{segment_counters[idx]}"
                segment_ids.append(segment_id)
                
                # Increment the counter for this passage
                segment_counters[idx] += 1
                
            # Decode input_ids back to text
            decoded_texts = self.tokenizer.batch_decode(
                tokenized["input_ids"], 
                skip_special_tokens=True  # Exclude special tokens
            )
            
            # Combine all chunks from all passages into a single list
            return {
                "passage_chunk_ids": segment_ids,
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "overflow_to_sample_mapping": tokenized["overflow_to_sample_mapping"],
                "decoded_texts": decoded_texts  # Add decoded texts to the output
            }
                
        hf_dataset = load_dataset(self.dataset_name, split='train')
        # Apply tokenization
        tokenized_dataset = hf_dataset.map(
            sliding_window_tokenization,
            batched=True,  # Process batches of samples
            batch_size=512,  # Adjust batch size based on memory
            num_proc=96,  # Multiprocessing for faster processing
            remove_columns=hf_dataset.column_names  # Remove original columns
        )
        
        return tokenized_dataset
    
    def get_tokenized_dataset(self):
        """
        Robustly handle dataset tokenization with caching.
        
        Args:
            embedder: The embedder object with a tokenize_dataset method
            self.tokenized_dataset_path (str): Path to save/load tokenized dataset
            self.force_retokenize (bool): Force re-tokenization even if cached dataset exists
        
        Returns:
            Tokenized dataset of HuggingFace Dataset type
        """
        # Check if cached dataset exists and we're not forcing retokenization
        if not self.force_retokenize and os.path.exists(self.tokenized_dataset_path):
            try:
                print(f"Loading previously tokenized dataset from {self.tokenized_dataset_path}")
                return load_from_disk(self.tokenized_dataset_path)
            except Exception as e:
                print(f"Error loading cached dataset: {e}. Proceeding with new tokenization.")
        
        # Tokenize the dataset
        print("Tokenizing dataset...")
        tokenized_dataset = self.tokenize_dataset()
        
        # Save the tokenized dataset
        try:
            tokenized_dataset.save_to_disk(self.tokenized_dataset_path)
            print(f"Tokenized dataset saved to {self.tokenized_dataset_path}")
        except Exception as e:
            print(f"Warning: Could not save tokenized dataset: {e}")
            
        return tokenized_dataset

    def prepare_dataloader(self, tokenized_dataset: Dataset, rank: int) -> DataLoader:
        """
        Prepare distributed dataloader
        
        Args:
            tokenized_dataset (Dataset): Tokenized input dataset
            rank (int): Current process rank
        
        Returns:
            DataLoader: Distributed dataloader
        """
        sampler = DistributedSampler(
            dataset=tokenized_dataset,
            num_replicas=self.world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
        
        dataloader = DataLoader(
            dataset=tokenized_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
            drop_last=False,
            sampler=sampler,
            collate_fn=self.custom_collate_fn
        )
        
        return dataloader

    def custom_collate_fn(self, batch):
        """
        Custom collate function to handle batching
        
        Args:
            batch (List): Batch of tokenized items
        
        Returns:
            Dict: Collated batch
        """
        return {
            "input_ids": torch.tensor([item["input_ids"] for item in batch]),
            "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
            "passage_chunk_ids": [item["passage_chunk_ids"] for item in batch],
            "decoded_texts": [item["decoded_texts"] for item in batch]
        }

    def generate_embeddings(self, rank: int, tokenized_dataset: Dataset):
        """
        Generate embeddings in a distributed manner
        
        Args:
            rank (int): Current process rank
            tokenized_dataset (Dataset): Tokenized input dataset
        """
        # Initialize process group
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=self.world_size)
        torch.cuda.set_device(rank)
        try:
            # Load Model
            model = AutoModel.from_pretrained(self.model_name).to(rank)
            
            # Prepare DataLoader
            dataloader = self.prepare_dataloader(tokenized_dataset, rank)
            
            all_embeddings = []
            all_passage_chunk_ids = []
            all_decoded_texts = []
            
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Rank {rank} Embedding Generation"):
                    # Move batch to GPU
                    input_ids = batch["input_ids"].to(rank)
                    attention_mask = batch["attention_mask"].to(rank)
                    
                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # Get CLS token embedding
                    embeddings = outputs.last_hidden_state[:, 0, :]
                    # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                    # Store embeddings (move to CPU and convert to numpy)
                    all_embeddings.append(embeddings.cpu().numpy())
                    all_passage_chunk_ids.extend(batch["passage_chunk_ids"])
                    all_decoded_texts.extend(batch["decoded_texts"])
                    

            # Combine embeddings
            all_embeddings = np.vstack(all_embeddings)
            
            # Save embeddings and passage ids
            # np.save(os.path.join(self.output_dir, f"miriad_4.4M_{self.model_name.split('/')[-1]}_{self.content}_embeddings_rank{rank}.npy"), all_embeddings)
            # np.save(os.path.join(self.output_dir, f"miriad_4.4M_{self.model_name.split('/')[-1]}_{self.content}_passage_chunk_ids_rank{rank}.npy"), all_passage_chunk_ids)
            # np.save(os.path.join(self.output_dir, f"miriad_4.4M_{self.model_name.split('/')[-1]}_{self.content}_decoded_texts_rank{rank}.npy"), all_decoded_texts)
            np.save(os.path.join(self.output_dir, f"miriad_{self.model_name.split('/')[-1]}_{self.content}_embeddings_rank{rank}.npy"), all_embeddings)
            np.save(os.path.join(self.output_dir, f"miriad_{self.model_name.split('/')[-1]}_{self.content}_passage_chunk_ids_rank{rank}.npy"), all_passage_chunk_ids)
            np.save(os.path.join(self.output_dir, f"miriad_{self.model_name.split('/')[-1]}_{self.content}_decoded_texts_rank{rank}.npy"), all_decoded_texts)
            
            print(f"Done saving to {self.output_dir}!")
            # # Optional: Upsert to Qdrant
            # if self.config.get('upsert_to_qdrant', False):
            #     self.upsert_to_qdrant(all_embeddings, all_passage_chunk_ids, all_decoded_texts)
        except Exception as e:
            print(f"Error in rank {rank}: {e}")
            
        finally:
            # Finalize process grou  p
            dist.destroy_process_group()

    def upsert_to_qdrant(self, embeddings: np.ndarray, passage_chunk_ids: List[str], decoded_texts: List[str]):
        """
        Upsert embeddings to Qdrant vector database
        
        Args:
            embeddings (np.ndarray): Numpy array of embeddings
            passage_ids (List[str]): Corresponding passage ids
            decoded_texts (List[str]): Decoded texts (z, question, answer, or qa)
        """
        qdrant_handler = QdrantHandlerPassageText(
            collection_name=self.config.get('qdrant_collection', 'passages'),
            host=self.config.get('qdrant_host', 'localhost'),
            port=self.config.get('qdrant_port', 6333)
        )
        
        
        qdrant_handler.upsert(embeddings, passage_chunk_ids, decoded_texts)

def main():
    # Argument parsing for configuration
    parser = argparse.ArgumentParser(description="Distributed Embedding Pipeline")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize embedder
    embedder = DistributedEmbedder(config)

    tokenized_dataset = embedder.get_tokenized_dataset()

    # Distributed embedding generation
    torch.multiprocessing.spawn(
        embedder.generate_embeddings, 
        args=(tokenized_dataset,), 
        nprocs=config.get('world_size', 8)
    )

if __name__ == "__main__":
    main()