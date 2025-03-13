import numpy as np
import threading
from gensim.downloader import load

# Load the model for query embedding in the main thread
global_model_thread = None

def load_model_for_thread():
    global global_model_thread
    if global_model_thread is None:
        global_model_thread = load("glove-wiki-gigaword-50")
    return global_model_thread

def compute_query_embedding(query):
    model = load_model_for_thread()
    words = query.split()
    vectors = []
    for word in words:
        if word in model:
            vectors.append(model[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

class DocumentRetriever:
    def __init__(self):
        self.similarities = []
        self.lock = threading.Lock()

    def _compute_similarity(self, query_embedding, embedding, index):
        sim = cosine_similarity(query_embedding, embedding)
        with self.lock:
            self.similarities.append((index, sim))

    def retrieve(self, query, embeddings, chunks, top_n=3):
        query_embedding = compute_query_embedding(query)
        threads = []
        self.similarities = []
        for index, emb in enumerate(embeddings):
            t = threading.Thread(target=self._compute_similarity, args=(query_embedding, emb, index))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        # Sort similarities in descending order and select top_n chunks
        self.similarities.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [(chunks[i], sim) for i, sim in self.similarities[:top_n]]
        return top_chunks
