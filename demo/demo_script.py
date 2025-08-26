"""
Demo script for showcasing the usage of embedded MIRIAD data
Assumes data is already embedded and upserted into qdrant_storage. 
"""

import sys
import yaml
import tqdm
from typing import Dict

sys.path.append("..")
from rag_pipeline.src.eval_utils import load_standardize_dataset
from rag_pipeline.src.rag import RAG
from rag_pipeline.src.printutils import pcprint, separator

def load_config(config_path = "../rag_pipeline/eval_config.yaml") -> Dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("Loaded config:")
    pcprint(config)
    return config

def get_rag_pipeline(config: Dict) -> RAG:
    embedding_model = config["embedding"]["model_name"]
    model_name_short = embedding_model.split("/")[-1]
    content = config["embedding"]["content"]
    collection_name = config["qdrant"]["collection"]\
            .format(model_name=model_name_short, content = content)
    
    return RAG(
            qdrant_host=config["qdrant"]["host"],
            qdrant_port=config["qdrant"]["port"],
            db_name=collection_name,
            embedding_model=embedding_model
            )

def test_retrieval(rag: RAG, query: str, top_k = 3):
    retrieved_points = rag(query, top_k)
    print(f"Top {top_k} QA pairs from MIRIAD that are most relevant to the query: {query}")
    for p in retrieved_points:
        print(p.payload["passage_text"])
        print("-" * 100, "\n")

def main():
    config = load_config()
    rag = get_rag_pipeline(config)
    test_retrieval(rag, "treatment for diabetes")

if __name__ == "__main__":
    main()
    
