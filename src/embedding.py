import numpy as np
from multiprocessing import Pool, cpu_count
import gensim.downloader as api

class EmbeddingCreator:
    """
    A generic class to compute embeddings for text chunks using different methods.
    
    Supported methods:
      - "glove": Uses pre-trained GloVe embeddings (via gensim) with multiprocessing.
      - "tfidf": Uses TfidfVectorizer from scikit-learn.
      - "bm25": Returns tokenized version of each chunk (for BM25 ranking).
    """
    
    def __init__(self, method="glove", model_name="glove-wiki-gigaword-50"):
        self.method = method.lower()
        self.model_name = model_name
        if self.method == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer()
            self.fitted = False
        elif self.method == "glove":
            # The model will be loaded in each worker via the initializer.
            pass
        elif self.method == "bm25":
            # BM25 will simply use tokenized text.
            pass
        else:
            raise ValueError("Unsupported method. Choose 'glove', 'tfidf', or 'bm25'.")

    def _init_worker(self):
        """Worker initializer: load the pre-trained GloVe model."""
        global _worker_model
        _worker_model = api.load(self.model_name)
    
    @staticmethod
    def _compute_glove_embedding(chunk):
        """Computes the embedding for a single chunk using the GloVe model."""
        global _worker_model
        words = chunk.split()
        vectors = [_worker_model[word] for word in words if word in _worker_model]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(_worker_model.vector_size)
    
    def create_embeddings(self, chunks):
        """
        Computes embeddings for each chunk based on the selected method.
        
        Parameters:
            chunks (list of str): The text chunks.
        
        Returns:
            For "glove": a numpy array of embeddings.
            For "tfidf": a numpy array of TF-IDF vectors.
            For "bm25": a list of tokenized chunks.
        """
        if self.method == "glove":
            with Pool(processes=cpu_count(), initializer=self._init_worker) as pool:
                embeddings = pool.map(EmbeddingCreator._compute_glove_embedding, chunks)
            return np.array(embeddings)
        elif self.method == "tfidf":
            self.fitted = True
            tfidf_matrix = self.vectorizer.fit_transform(chunks)
            return tfidf_matrix.toarray()
        elif self.method == "bm25":
            # For BM25, tokenize each chunk (optionally create bigrams)
            tokenized = [chunk.split() for chunk in chunks]
            return tokenized
        else:
            raise ValueError("Unsupported method. Choose 'glove', 'tfidf', or 'bm25'.")
