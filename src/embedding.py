import numpy as np
from multiprocessing import Pool, cpu_count
import gensim.downloader as api
from src.logger import Logger

class EmbeddingException(Exception):
    """Custom exception for embedding creation errors."""
    pass

class EmbeddingCreator:
    """
    Creates embeddings for text chunks using multiprocessing.
    Supports 'glove' and 'dense' methods, where 'dense' is treated as an alias for 'glove' in this implementation.
    """

    def __init__(self, method="glove", model_name="glove-wiki-gigaword-50"):
        """
        Parameters:
            method (str): The embedding method. Accepts "glove" or "dense".
            model_name (str): Name of the pre-trained model to load.
        """
        self.method = method.lower()
        self.logger = Logger("logs/embedding.log", verbose=False)
        if self.method in ["glove", "dense"]:
            try:
                self.word_model = api.load(model_name)
                self.logger.log(f"Model '{model_name}' loaded successfully with method '{self.method}'.", level="INFO")
            except Exception as e:
                raise EmbeddingException(f"Failed to load model '{model_name}': {e}")
        else:
            raise ValueError("Invalid method. Choose 'glove' (or 'dense') for embedding creation.")

    def _compute_embedding(self, chunk):
        """
        Computes the embedding for a single chunk by averaging its word vectors.
        This is the worker function for multiprocessing.
        """
        try:
            words = chunk.split()
            vectors = [self.word_model[word] for word in words if word in self.word_model]
            if vectors:
                return np.mean(vectors, axis=0)
            else:
                # If no word vectors are found, return a zero vector.
                return np.zeros(self.word_model.vector_size)
        except Exception as e:
            raise EmbeddingException(f"Failed to compute embedding for chunk: {e}")

    def create_embeddings(self, chunks):
        """
        Computes embeddings in parallel for all chunks.
        
        Parameters:
            chunks (list): List of text chunks.
        
        Returns:
            np.array: An array of embeddings.
        """
        if not chunks:
            self.logger.log("No chunks provided for embedding creation.", level="WARNING")
            return np.array([])
        try:
            with Pool(processes=cpu_count()) as pool:
                embeddings = pool.map(self._compute_embedding, chunks)
            self.logger.log(f"Created embeddings for {len(chunks)} chunks using multiprocessing.", level="INFO")
            return np.array(embeddings)
        except Exception as e:
            self.logger.error(f"Error in multiprocessing embedding creation: {e}")
            return np.array([])
