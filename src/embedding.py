import numpy as np
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

class EmbeddingCreator:
    """Creates embeddings for text chunks using sentence transformers."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, chunks):
        """Computes embeddings in parallel for all chunks."""
        with ThreadPoolExecutor() as executor:
            embeddings = list(executor.map(self.model.encode, chunks))
        return np.array(embeddings)
