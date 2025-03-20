import numpy as np
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from src.text_similarity import TextSimilarity
from src.logger import Logger  # Import Logger

class TextRetriever:
    """Retrieves and stores relevant text chunks using BM25, TF-IDF, or GloVe similarity in a multi-threaded way."""

    def __init__(self, method="GloVe", db_path="db/retrieval_results.db"):
        """
        Parameters:
            method (str): Similarity method ('bm25', 'tfidf', or 'word').
            db_path (str): Path to the SQLite database for storing results.
        """
        self.similarity_calculator = TextSimilarity(method=method)
        self.method = method.lower()
        self.db_path = db_path
        self.chunks = None
        self.logger = Logger("retrieval.log")  # Initialize Logger
        self.lock = threading.Lock()  # Ensures thread-safe database writes
        self.setup_database()
        self.logger.log(f"TextRetriever initialized with method: {self.method}", level="INFO")

    def setup_database(self):
        """Creates an SQLite table to store retrieval results if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS retrieval_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    chunk TEXT,
                    score REAL
                )
            ''')
            conn.commit()
            conn.close()
            self.logger.log("Database setup complete.", level="INFO")
        except Exception as e:
            self.logger.error(f"Database setup failed: {e}")

    def normalize_scores(self, scores, method="log"):
        """Applies different normalization methods: min-max, log scaling, or soft normalization."""
        try:
            min_score, max_score = min(scores), max(scores)
            if max_score == min_score:
                return [0.5] * len(scores)
            if method == "min-max":
                return [(score - min_score) / (max_score - min_score) for score in scores]
            elif method == "soft":
                return [(score - min_score) / (max_score - min_score + 1e-6) for score in scores]
            elif method == "log":
                log_scores = np.log1p(scores)
                min_log, max_log = min(log_scores), max(log_scores)
                return [(s - min_log) / (max_log - min_log) for s in log_scores]
            else:
                raise ValueError("Invalid normalization method. Choose 'min-max', 'soft', or 'log'.")
        except Exception as e:
            self.logger.error(f"Normalization failed: {e}")
            return scores

    def store_results(self, query, results):
        """Stores normalized retrieval results in SQLite using a thread lock to prevent conflicts."""
        try:
            with self.lock:  # Ensure thread safety while writing to DB
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                for chunk, score in results:
                    cursor.execute(
                        "INSERT INTO retrieval_results (query, chunk, score) VALUES (?, ?, ?)", 
                        (query, chunk, score)
                    )
                conn.commit()
                conn.close()
            self.logger.log(f"Results stored for query: {query}", level="INFO")
        except Exception as e:
            self.logger.error(f"Failed to store results for query '{query}': {e}")

    def _compute_similarity(self, query, chunk):
        """Computes similarity for a given chunk in a thread-safe manner."""
        try:
            return self.similarity_calculator.compute_similarity(query, chunk)
        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            return 0.0

    def retrieve_top_chunks(self, query, chunks, top_k=3, normalization_method="log"):
        """Finds the most relevant text chunks using multi-threaded similarity computation.

        Parameters:
            query (str): The query string.
            chunks (list): List of text chunks.
            top_k (int): Number of top chunks to return.
            normalization_method (str): Normalization method ('min-max', 'soft', or 'log').

        Returns:
            List of tuples (chunk, normalized_similarity_score).
        """
        self.chunks = chunks
        self.logger.log(f"Starting retrieval for query: {query} using {self.method.upper()}", level="INFO")

        try:
            if self.method == "bm25":
                # Fit BM25 on the entire corpus of chunks
                self.similarity_calculator.fit_bm25(chunks)
                # Compute BM25 scores for all chunks using the query tokens
                query_tokens = self.similarity_calculator.tokenize_with_bigrams(query)
                raw_scores = self.similarity_calculator.bm25.get_scores(query_tokens)
            else:
                if self.method == "tfidf":
                    self.similarity_calculator.fit_tfidf(chunks)
                
                # Compute similarity in parallel using ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    raw_scores = list(executor.map(self._compute_similarity, [query] * len(chunks), chunks))

            normalized_scores = self.normalize_scores(raw_scores, method=normalization_method)
            ranked_indices = np.argsort(normalized_scores)[::-1][:top_k]
            results = [(chunks[i], normalized_scores[i]) for i in ranked_indices]

            self.logger.log(f"Top {top_k} chunks retrieved for query: {query}", level="INFO")
            self.store_results(query, results)

            return results
        except Exception as e:
            self.logger.error(f"Retrieval failed for query '{query}': {e}")
            return []
