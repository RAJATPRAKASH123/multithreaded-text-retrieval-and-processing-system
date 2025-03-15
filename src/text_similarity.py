import numpy as np
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from scipy.spatial.distance import cosine

class TextSimilarity:
    """Computes similarity using BM25, TF-IDF, or Word Embeddings (GloVe)."""

    def __init__(self, method="bm25"):
        """
        :param method: "word" (GloVe), "tfidf" (TF-IDF), or "bm25" (BM25).
        """
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
        else:
            raise ValueError("Invalid method. Choose 'word', 'tfidf', or 'bm25'.")

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
