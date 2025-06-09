import torch

class Embedder:
    def __init__(
        self,
        embedding_type,
        embedding_model="all-mpnet-base-v2",
        embedding_format="questions",
    ):
        self.embedding_format = embedding_format

        if embedding_type == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            if torch.cuda.is_available():
                self.embedder = SentenceTransformer(embedding_model, device="cuda")
            else:
                self.embedder = SentenceTransformer(embedding_model, device="cpu")
            self.embed_fn = self._embed_sentencetransformers
        elif embedding_type == "medcpt":
            try:
                from src import embed_medcpt
            except:
                import embed_medcptx
            self.embed_fn = self._embed_medcpt
            self.embedder = embed_medcpt.MedCPT()

    def __call__(self, data):
        """['qa_id', 'paper_id', 'question', 'answer', 'paper_url', 'paper_title', 'passage_text', 'passage_position', 'year']"""
        # input_text = [self.format_embedding(element) for element in batch]
        formatted_data = self._format_embedding(data)
        return self.embed_fn(formatted_data)

    def _embed_sentencetransformers(self, input_text):
        """
        input_text could be a string or a list of strings (passage chunks).
        """
        embeddings = self.embedder.encode(input_text, convert_to_tensor=True)
        if self.embedding_format == "passage_text_chunks" and len(embeddings.shape) > 1:
            embeddings = embeddings.mean(axis=0)
        return embeddings

    def _embed_medcpt(self, input_text):
        return self.embedder.embed(input_text)
    
    def _format_embedding(self, data):
        if self.embedding_format == "questions":
            return data["question"]
        elif self.embedding_format == "questions-answers":
            return data["question"] + " " + data["answer"]
        elif self.embedding_format == "passage_text_chunks":
            return data["passage_text_chunks"]


def format_embeddings(data, embedding_format):
    if embedding_format == "questions":
        return data["question"]
    elif embedding_format == "questions-answers":
        return data["question"] + " " + data["answer"]
    elif embedding_format == "passage_text_chunks":
        return data["passage_text_chunks"]
