import numpy as np
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from scipy.spatial.distance import cosine
from src.logger import Logger
from src.embedding import EmbeddingException


class TextSimilarity:
    """Computes similarity using BM25, TF-IDF, or Word Embeddings (GloVe)."""

    def __init__(self, method="bm25", model_name="glove-wiki-gigaword-50"):
        """
        :param method: "word" (GloVe), "tfidf" (TF-IDF), "bm25" (BM25), or "dense" (alias for glove).
        :param model_name: Name of the pre-trained model to load.
        """
        self.logger = Logger("logs/text_similarity.log", verbose=False)
        self.method = method.lower()
        if self.method == "word":
            self.word_model = api.load("glove-wiki-gigaword-50")
        elif self.method == "tfidf":
            self.vectorizer = TfidfVectorizer()
            self.fitted_matrix = None
            self.sentences = []
        elif self.method == "bm25":
            self.bm25 = None
            self.sentences = []
        elif self.method == "dense":
            try:
                self.word_model = api.load(model_name)
                self.logger.log(f"Model '{model_name}' loaded successfully with method '{self.method}'.", level="INFO")
            except Exception as e:
                raise EmbeddingException(f"Failed to load model '{model_name}': {e}")
        else:
            raise ValueError("Invalid method. Choose 'word', 'tfidf', 'bm25', or 'dense'.")
        self.logger.log(f"TextSimilarity initialized with method: {self.method}", level="INFO")

    def tokenize_with_bigrams(self, text):
        """Tokenizes text into words and bigrams."""
        words = text.split()
        bigrams = ["_".join(pair) for pair in zip(words[:-1], words[1:])]
        return words + bigrams

    def fit_tfidf(self, corpus):
        """Fits TF-IDF on the entire chunk corpus."""
        self.sentences = self.filter_chunks(corpus)
        self.fitted_matrix = self.vectorizer.fit_transform(self.sentences).toarray()

    def transform_tfidf(self, text):
        """Transforms a query sentence using the trained TF-IDF vectorizer."""
        return self.vectorizer.transform([text]).toarray()[0]

    def fit_bm25(self, corpus, k1=1.5, b=0.75):
        """Fits BM25 with bigram tokenization and parameter tuning."""
        self.sentences = [self.tokenize_with_bigrams(text) for text in self.filter_chunks(corpus)]
        self.bm25 = BM25Okapi(self.sentences, k1=k1, b=b)

    def compute_bm25_similarity(self, query):
        """Computes BM25 similarity scores with query term weighting."""
        query_tokens = self.tokenize_with_bigrams(query)
        scores = self.bm25.get_scores(query_tokens)

        # Boost query-related terms for better ranking
        boosted_scores = [score * 1.2 if token in query_tokens else score for score, token in zip(scores, self.sentences)]
        return np.mean(boosted_scores)


    def compute_word_embedding(self, text):
        """Averages word vectors to create a sentence embedding."""
        words = text.split()
        vectors = [self.word_model[word] for word in words if word in self.word_model]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.word_model.vector_size)

    def compute_similarity(self, text1, text2):
        """Computes similarity between two texts using the selected method."""
        if self.method == "word":
            emb1, emb2 = self.compute_word_embedding(text1), self.compute_word_embedding(text2)
            similarity = 1 - cosine(emb1, emb2)
        elif self.method == "tfidf":
            emb1, emb2 = self.transform_tfidf(text1), self.transform_tfidf(text2)
            similarity = 1 - cosine(emb1, emb2)
        elif self.method == "bm25":
            similarity = self.compute_bm25_similarity(text1)
        else:
            raise ValueError("Invalid method. Choose 'word', 'tfidf', or 'bm25'.")
        return max(0, similarity)

    def filter_chunks(self, chunks):
        """Removes generic or low-information chunks."""
        stop_words = {"see also", "references", "external links", "further reading"}
        return [
            chunk for chunk in chunks 
            if len(chunk.split()) > 10 and not any(word in chunk.lower() for word in stop_words)
        ]

def cosine_similarity(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return 1 - cosine(vec1, vec2)