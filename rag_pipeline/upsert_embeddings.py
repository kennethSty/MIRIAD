import os
import json
import uuid
import numpy as np
import time
import glob
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, Distance, VectorParams
from src.qdrant_utils import str2bool

import argparse




class CheckpointManager:
    def __init__(self, checkpoint_dir: str = 'upsert_checkpoints'):
        """
        Manage checkpoints for long-running upsert operations
        
        Args:
            checkpoint_dir (str): Directory to store checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self, 
        embedding_file: str, 
        processed_indices: List[int], 
        metadata: Dict[str, Any] = None
    ):
        """
        Save a checkpoint for a specific embedding file
        """
        checkpoint_filename = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_{os.path.basename(embedding_file)}.json"
        )
        
        checkpoint_data = {
            "file": embedding_file,
            "processed_indices": processed_indices,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        with open(checkpoint_filename, 'w') as f:
            json.dump(checkpoint_data, f)

    def load_checkpoint(self, embedding_file: str) -> Optional[Dict]:
        """
        Load checkpoint for a specific embedding file
        """
        checkpoint_filename = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_{os.path.basename(embedding_file)}.json"
        )
        
        if os.path.exists(checkpoint_filename):
            with open(checkpoint_filename, 'r') as f:
                return json.load(f)
        return None

    def cleanup_checkpoint(self, embedding_file: str):
        """
        Remove checkpoint file after successful completion
        """
        checkpoint_filename = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_{os.path.basename(embedding_file)}.json"
        )
        
        if os.path.exists(checkpoint_filename):
            os.remove(checkpoint_filename)



def process_embedding_file(
    file_path: str, 
    collection_name: str,
    vector_size: int,
    host: str = 'localhost',
    port: int = 6333,
    checkpoint_dir: str = 'upsert_checkpoints',
    batch_size: int = 1000,
    check_if_exists: bool = True,
    resume: bool = True,
    overwrite_existing_collection: bool = False
):
    """
    Process a single embedding file with checkpointing
    
    Args:
        file_path (str): Path to the embedding file
        collection_name (str): Qdrant collection name
        vector_size (int): Size of embedding vectors
        host (str): Qdrant server host
        port (int): Qdrant server port
        checkpoint_dir (str): Directory for checkpoint files
        batch_size (int): Number of points to upsert in a batch
        check_if_exists (bool): Whether to check for existing points
        resume (bool): Whether to resume from last checkpoint
        overwrite_existing_collection (bool): Whether to overwrite existing collection with a complet new one
    """
    # Ensure collection exists with proper configuration
    # Create separate client for this process
    client = QdrantClient(host=host, port=port)
    
    def check_points_exist(
        payload_keys: List[str], 
        payload_values: List[Any]
    ) -> List[bool]:
        """
        Check existence of points efficiently using Qdrant's filter
        Returns list of boolean indicating existence for each input
        """
        # Construct filter for multiple payload checks
        filters = [{
            "must": [
                {"key": key, "match": {"value": value}}
                for key, value in zip(payload_keys, payload_values)
            ]
        }]

        # Count matching points
        count = client.count(
            collection_name=collection_name,
            count_filter=filters[0]
        )

        return [count.count > 0]
    # Ensure collection exist (centralied collection creation in create_collection)
    print(f"Collection {collection_name} exists: {client.collection_exists(collection_name)}")
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir)

    # Load checkpoint if resuming
    checkpoint = None
    start_index = 0
    if resume:
        checkpoint = checkpoint_manager.load_checkpoint(file_path)
        if checkpoint:
            start_index = max(checkpoint['processed_indices'])
            print(f"Resuming from checkpoint at index {start_index}")
            
    # Load numpy files
    embeddings = np.load(file_path)
    passage_chunk_ids = np.load(file_path.replace('embeddings', 'passage_chunk_ids'))
    texts = np.load(file_path.replace('embeddings', 'decoded_texts'))

    # Slice arrays from the last checkpoint
    embeddings = embeddings[start_index:]
    passage_chunk_ids = passage_chunk_ids[start_index:]
    texts = texts[start_index:]

    points_to_upsert = []
    processed_indices = []
    skipped_points = 0
    try:
        for i, (emb, passage_chunk_id, text) in enumerate(
            zip(embeddings, passage_chunk_ids, texts), 
            start=start_index
        ):
            # Optional: Check point existence
            if check_if_exists:
                # Efficient existence check
                exists = check_points_exist(
                    ['passage_chunk_id'], 
                    [passage_chunk_id]
                )[0]
                
                if exists:
                    skipped_points += 1
                    continue
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={
                    "passage_chunk_id": passage_chunk_id,
                    "passage_text": text,
                }
            )
            
            points_to_upsert.append(point)
            processed_indices.append(i)
            
            # Batch upsert
            if len(points_to_upsert) >= batch_size:
                client.upsert(
                    collection_name=collection_name, 
                    points=points_to_upsert,
                    wait=True
                )
                print(f"Processed {file_path}: Upserted {len(points_to_upsert)} (this batch), Skipped {skipped_points} points (in total)")
                
                # Create checkpoint
                checkpoint_manager.save_checkpoint(
                    file_path, 
                    processed_indices,
                    metadata={
                        "total_processed": len(processed_indices),
                        "last_point_id": point.id
                    }
                )
                print(f"Processed {file_path}: Upserted {len(points_to_upsert)} (this batch), Skipped {skipped_points} points (in total)")
                
                # Reset for next batch
                points_to_upsert = []
                processed_indices = []

                
        # Upsert any remaining points
        if points_to_upsert:
            client.upsert(
                collection_name=collection_name, 
                points=points_to_upsert,
                wait=True
            )
            print(f"Processed {file_path}: Final upsert for {len(points_to_upsert)} (this batch), Skipped {skipped_points} points (in total)")

        # Remove checkpoint on successful completion
        checkpoint_manager.cleanup_checkpoint(file_path)
        
    except Exception as e: 
        print(f"Error processing {file_path}")
        raise e

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Qdrant upserts for embedding files.")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the embedding file")
    parser.add_argument('--collection_name', type=str, required=True, help="Qdrant collection name")
    parser.add_argument('--vector_size', type=int, required=True, help="Size of embedding vectors")
    parser.add_argument('--host', type=str, default='localhost', help="Qdrant server host")
    parser.add_argument('--port', type=int, default=6333, help="Qdrant server port")
    parser.add_argument('--checkpoint_dir', type=str, default='upsert_checkpoints', help="dir to store checkpoints")
    parser.add_argument('--batch_size', type=int, default=1000, help="Batch size for upserts")
    parser.add_argument('--check_if_exists', type=str2bool, default=False, help="Check if points already exist")
    parser.add_argument('--resume', type=str2bool, default=True, help="Resume from last checkpoint")
    parser.add_argument('--overwrite_existing_collection', type=str2bool, default=False, help="Overwrite existing collection")

    args = parser.parse_args()
    
    # Call the function with parsed arguments
    process_embedding_file(
        file_path=args.file_path,
        collection_name=args.collection_name,
        vector_size=args.vector_size,
        host=args.host,
        port=args.port,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        check_if_exists=args.check_if_exists,
        resume=args.resume,
        overwrite_existing_collection=args.overwrite_existing_collection,
    )
    
if __name__ == "__main__":
    main()