from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
import time
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict
import numpy as np


class RAG:
    def __init__(self, qdrant_host, qdrant_port, db_name, embedding_model):
        self.client = QdrantClient(qdrant_host, port=qdrant_port)
        self.collection_name = db_name
        self.embedding_model = embedding_model
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        self.model = AutoModel.from_pretrained(self.embedding_model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def __call__(self, query, topk=5):
        embedding = self.get_embeddings(query)
        has_error = False
        for _ in range(3):
            try:
                search_result = self.client.search(
                    collection_name=self.collection_name, query_vector=embedding[0], limit=topk,
                    timeout=240,
                )
                if has_error:
                    print("Recovery successful")
                return search_result
            except Exception as e:
                print(f"Error during search: {e}")
                has_error = True
                time.sleep(30)
        return None
    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Get embeddings for one text sequence
        """
        all_embeddings = []
            
        # Tokenize text sequence
        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Take the [CLS] token embedding (first token)
        embeddings = model_output.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]
        # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        all_embeddings=embeddings.cpu().numpy()
        
        return all_embeddings

class BatchRAG:
    def __init__(self, qdrant_host, qdrant_port, db_name, embedding_model):
        self.client = QdrantClient(qdrant_host, port=qdrant_port)
        self.collection_name = db_name
        self.embedding_model = embedding_model
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        self.model = AutoModel.from_pretrained(self.embedding_model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def get_embeddings(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Get embeddings for a list of texts in batches
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Take the [CLS] token embedding (first token)
            embeddings = model_output.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]
            print(f"Embeddings shape: {embeddings.shape}")
            # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
            
        return np.vstack(all_embeddings)
    
    def batch_query(self, questions: List[str], topk: int = 3, batch_size: int = 64) -> List[List[Dict]]:
        """
        Batch process questions to get nearest neighbors
        """
        has_error = False
        # Get embeddings for all questions
        embeddings = self.get_embeddings(questions, batch_size=batch_size)
        
        # Search in batches
        all_results = []
        for embedding in embeddings:
            for _ in range(3):
                try:
                    search_result = self.client.search(
                        collection_name=self.collection_name, query_vector=embedding, limit=topk,
                        timeout=240,
                    )
                    if has_error:
                        print("Recovery successful")
                    break
                except Exception as e:
                    print(f"Error during search: {e}")
                    has_error = True
                    time.sleep(30)
                    search_result = None
            all_results.append(search_result)
            
        return all_results
