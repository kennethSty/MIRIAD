import sys
import yaml
from typing import Dict
from pathlib import Path

# Add the parent directories to path to import rag_pipeline
#current_dir = Path(__file__).parent.absolute()
#rag_pipeline_path = current_dir.parent.parent.parent / "rag_pipeline"
#sys.path.insert(0, str(rag_pipeline_path / "src"))
from rag_pipeline.src.rag import RAG


class MiriadRetriever:
    """Retriever that uses the MIRIAD RAG pipeline to retrieve relevant documents."""
    
    def __init__(self, config):
        self.config = config
        self.retriever = self._setup_retriever()

    def retrieve_docs(self, query: str, top_k: int):
        """Retrieve documents using the RAG retriever.
        
        Args:
            query: Search query string
            top_k: Number of top documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        return self.retriever(query, top_k)

    def _setup_retriever(self):
        """Setup the RAG retriever with configuration."""
            
        embedding_model = self.config["embedding"]["emb_model_name"]
        model_name_short = embedding_model.split("/")[-1]
        content = self.config["embedding"]["content"]
        collection_name = self.config["qdrant"]["collection"].format(
            emb_model_name=model_name_short, content=content
        )

        return RAG(
            qdrant_host=self.config["qdrant"]["host"],
            qdrant_port=self.config["qdrant"]["port"],
            db_name=collection_name,
            embedding_model=embedding_model
        )
