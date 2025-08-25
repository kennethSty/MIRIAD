from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from src.qdrant_utils import str2bool
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Qdrant collection createion for embedding files.")
    parser.add_argument('--collection_name', type=str, required=True, help="Qdrant collection name")
    parser.add_argument('--vector_size', type=int, required=True, help="Size of embedding vectors")
    parser.add_argument('--host', type=str, default='localhost', help="Qdrant server host")
    parser.add_argument('--port', type=int, default=6333, help="Qdrant server port")
    parser.add_argument('--checkpoint_dir', type=str, default='upsert_checkpoints', help="dir to store checkpoints")
    parser.add_argument('--check_if_exists', type=str2bool, default=False, help="Check if points already exist")
    parser.add_argument('--overwrite_existing_collection', type=str2bool, default=False, help="Overwrite existing collection")
    args = parser.parse_args()
    
    client = QdrantClient(host=args.host, port=args.port)
    
    if client.collection_exists(args.collection_name) and not args.overwrite_existing_collection:
        print(f"Collection '{args.collection_name}' exists! Skipped creation.")
        return
    
    elif not client.collection_exists(args.collection_name):
        client.create_collection(
            collection_name=args.collection_name,
            vectors_config=VectorParams(
                    size=args.vector_size, 
                    distance=Distance.COSINE,
                    on_disk=True
                ),
            hnsw_config=models.HnswConfigDiff(on_disk=True)
        )
    elif args.overwrite_existing_collection:
        client.delete_collection(collection_name=args.collection_name)
        client.create_collection(
            collection_name=args.collection_name,
            vectors_config=VectorParams(
                    size=args.vector_size, 
                    distance=Distance.COSINE,
                    on_disk=True
                ),
            hnsw_config=models.HnswConfigDiff(on_disk=True)
        )
    print(f"Created collection {args.collection_name}!")
    return

if __name__ == "__main__":
    main()