import numpy as np
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine

class SimilarityCalculator:
    """Computes similarity using either word embeddings (GloVe) or sentence-level TF-IDF."""

    def __init__(self, method="word"):
        """
        :param method: "word" for word embeddings (GloVe), "sentence" for TF-IDF-based similarity.
        """
        self.method = method.lower()

        if self.method == "word":
            self.word_model = api.load("glove-wiki-gigaword-50")  # Load GloVe embeddings
        elif self.method == "sentence":
            self.vectorizer = TfidfVectorizer()  # Initialize TF-IDF but don't fit it yet
            self.fitted_matrix = None  # Placeholder for fitted TF-IDF matrix
            self.sentences = []  # Store sentences for fitting
        else:
            raise ValueError("Invalid method. Choose 'word' or 'sentence'.")

    def fit_sentence_embeddings(self, corpus):
        """Fits TF-IDF vectorizer on the entire chunk corpus."""
        self.sentences = corpus
        self.fitted_matrix = self.vectorizer.fit_transform(corpus).toarray()

    def transform_sentence_embedding(self, text):
        """Transforms a new query sentence using the fitted TF-IDF vectorizer."""
        return self.vectorizer.transform([text]).toarray()[0]

    def compute_word_embedding(self, text):
        """Averages word vectors to create a sentence embedding."""
        words = text.split()
        vectors = [self.word_model[word] for word in words if word in self.word_model]

        if not vectors:
            return np.zeros(self.word_model.vector_size)  # Handle missing words

        return np.mean(vectors, axis=0)

    def compute_similarity(self, text1, text2):
        """Computes similarity between two texts using the selected method."""
        if self.method == "word":
            emb1, emb2 = self.compute_word_embedding(text1), self.compute_word_embedding(text2)
        elif self.method == "sentence":
            emb1 = self.transform_sentence_embedding(text1)
            emb2 = self.transform_sentence_embedding(text2)
        else:
            raise ValueError("Invalid method. Choose 'word' or 'sentence'.")

        # Compute cosine similarity
        similarity = 1 - cosine(emb1, emb2)
        return max(0, similarity)  # Ensure non-negative similarity
