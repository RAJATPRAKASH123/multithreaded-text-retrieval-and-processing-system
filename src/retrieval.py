import numpy as np
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from src.text_similarity import TextSimilarity

class TextRetriever:
    """Retrieves and stores relevant text chunks using BM25, TF-IDF, or GloVe similarity."""

    def __init__(self, method="GloVe", db_path="db/retrieval_results.db"):
        self.similarity_calculator = TextSimilarity(method=method)
        self.method = method.lower()
        self.db_path = db_path
        self.chunks = None
        self.setup_database()

    def setup_database(self):
        """Creates SQLite table to store retrieval results if not exists."""
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

    def normalize_scores(self, scores, method="log"):
        """Applies different normalization methods: min-max, log scaling, or soft normalization."""
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

    def store_results(self, query, results):
        """Stores normalized retrieval results in SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for chunk, score in results:
            cursor.execute("INSERT INTO retrieval_results (query, chunk, score) VALUES (?, ?, ?)", (query, chunk, score))
        conn.commit()
        conn.close()

    def retrieve_top_chunks(self, query, chunks, top_k=3, normalization_method="log"):
        """Finds most relevant text chunks using similarity computation.
        
        For BM25, the method computes scores for all chunks at once.
        For TF-IDF or word-based methods, it computes each chunk's score individually.
        
        Parameters:
            query (str): The query string.
            chunks (list): List of text chunks.
            top_k (int): Number of top chunks to return.
            normalization_method (str): Normalization method ('min-max', 'soft', or 'log').
            
        Returns:
            List of tuples (chunk, normalized_similarity_score).
        """
        self.chunks = chunks

        if self.method == "bm25":
            # Fit BM25 on the entire corpus of chunks
            self.similarity_calculator.fit_bm25(chunks)
            # Compute BM25 scores for all chunks using the query tokens
            query_tokens = self.similarity_calculator.tokenize_with_bigrams(query)
            raw_scores = self.similarity_calculator.bm25.get_scores(query_tokens)
        else:
            if self.method == "tfidf":
                self.similarity_calculator.fit_tfidf(chunks)
            # For word or tfidf, compute similarity for each chunk individually
            def compute_similarity(chunk):
                return self.similarity_calculator.compute_similarity(query, chunk)
            with ThreadPoolExecutor() as executor:
                raw_scores = list(executor.map(compute_similarity, chunks))

        normalized_scores = self.normalize_scores(raw_scores, method=normalization_method)
        ranked_indices = np.argsort(normalized_scores)[::-1][:top_k]
        results = [(chunks[i], normalized_scores[i]) for i in ranked_indices]
        self.store_results(query, results)
        return results
