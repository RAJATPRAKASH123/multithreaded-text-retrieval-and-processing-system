import os
import numpy as np
import logging
from multiprocessing import Pool, cpu_count
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from src.logger import Logger  # Custom Logger Class
from src.exceptions import EmbeddingException  # Custom Exception Class

class EmbeddingCreator:
    """Creates embeddings for text chunks using BM25, TF-IDF, or GloVe similarity with multiprocessing."""
    
    def __init__(self, method="bm25", model_name=None):
        """
        Parameters:
            method (str): The embedding method ('bm25', 'tfidf', or 'glove').
            model_name (str, optional): Path to the GloVe model file if using GloVe.
        """
        self.method = method.lower()
        self.model_name = model_name
        self.vectorizer = None
        self.logger = Logger("embedding.log")  # Logger instance


    def _init_worker(self, chunks):
        """Worker initialization: initializes BM25 or TF-IDF model for multiprocessing."""
        global vectorizer
        try:
            if self.method == "bm25":
                tokenized_chunks = [chunk.split() for chunk in chunks]
                vectorizer = BM25Okapi(tokenized_chunks)
                self.logger.log("BM25 model initialized successfully.", level="INFO")
            elif self.method == "tfidf":
                vectorizer = TfidfVectorizer()
                vectorizer.fit(chunks)
                self.logger.log("TF-IDF model initialized successfully.", level="INFO")
        except Exception as e:
            raise EmbeddingException(f"Failed to initialize {self.method.upper()} model: {e}")

    @staticmethod
    def _compute_bm25_embedding(chunk):
        """Computes BM25 scores for the given chunk."""
        global vectorizer
        try:
            return np.array(vectorizer.get_scores(chunk.split()))
        except Exception as e:
            Logger("embedding.log").error(f"BM25 embedding failed: {e}")
            raise EmbeddingException(f"BM25 embedding failed: {e}")

    @staticmethod
    def _compute_tfidf_embedding(chunk):
        """Computes TF-IDF vector for the given chunk."""
        global vectorizer
        try:
            return np.array(vectorizer.transform([chunk]).toarray().flatten())
        except Exception as e:
            Logger("embedding.log").error(f"tfidf embedding failed: {e}")
            raise EmbeddingException(f"TF-IDF embedding failed: {e}")

    @staticmethod
    def _compute_glove_embedding(chunk):
        """Computes GloVe embeddings (averaging word vectors)."""
        global glove_model
        try:
            words = chunk.split()
            word_vectors = [glove_model[word] for word in words if word in glove_model]
            if not word_vectors:
                return np.zeros(300)
            return np.mean(word_vectors, axis=0)
        except Exception as e:
            Logger("embedding.log").error(f"GloVe embedding failed: {e}")
            raise EmbeddingException(f"GloVe embedding failed: {e}")

    def create_embeddings(self, chunks):
        """Computes embeddings in parallel for BM25, TF-IDF, or GloVe."""
        if not chunks:
            self.logger.log("No chunks provided for embedding.", level="WARNING")
            return np.array([])

        try:
            with Pool(processes=cpu_count(), initializer=self._init_worker, initargs=(chunks,)) as pool:
                if self.method == "bm25":
                    embeddings = pool.map(self._compute_bm25_embedding, chunks)
                elif self.method == "tfidf":
                    embeddings = pool.map(self._compute_tfidf_embedding, chunks)
                elif self.method == "glove":
                    global glove_model
                    self.logger.log(f"Loading GloVe model from {self.model_name}...", "INFO")
                    glove_model = KeyedVectors.load_word2vec_format(self.model_name, binary=False)
                    embeddings = pool.map(self._compute_glove_embedding, chunks)
                else:
                    raise EmbeddingException(f"Invalid embedding method: {self.method}")

            self.logger.log(f"{self.method.upper()} embeddings created successfully.", level="INFO")
            return np.array(embeddings)

        except EmbeddingException as e:
            self.logger.error(f"Embedding creation failed: {e}")
            raise e  # Raising the exception after logging

        except Exception as e:
            self.logger.error(f"Unexpected error during embedding: {e}")
            raise EmbeddingException(f"Unexpected embedding error: {e}")

