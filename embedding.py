import gensim.downloader as api
import numpy as np
from multiprocessing import Pool, cpu_count

# Global variable for the model in worker processes
global_model = None

def init_worker():
    global global_model
    global_model = api.load("glove-wiki-gigaword-50")

def compute_embedding(chunk):
    global global_model
    words = chunk.split()
    vectors = []
    for word in words:
        if word in global_model:
            vectors.append(global_model[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(global_model.vector_size)

class EmbeddingCreator:
    def __init__(self):
        # The model is loaded within each worker process via init_worker
        pass

    def compute_embeddings(self, chunks):
        with Pool(cpu_count(), initializer=init_worker) as pool:
            embeddings = pool.map(compute_embedding, chunks)
        return embeddings
