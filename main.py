from extraction import DataExtractor
from embedding import EmbeddingCreator
from retrieval import DocumentRetriever
from processing import process_chunks
import asyncio

def main():
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    query = "What is the impact of AI?"
    
    # Data Extraction and Cleaning
    extractor = DataExtractor(url)
    print("Extracting and cleaning data...")
    chunks = extractor.extract_and_process()
    print(f"Extracted {len(chunks)} chunks.\n")
    
    # Embedding Creation using Multiprocessing
    print("Computing embeddings for chunks...")
    embed_creator = EmbeddingCreator()
    embeddings = embed_creator.compute_embeddings(chunks)
    print("Embeddings computed.\n")
    
    # Document Retrieval using Threading
    retriever = DocumentRetriever()
    print("Retrieving relevant chunks...")
    top_chunks = retriever.retrieve(query, embeddings, chunks)
    for idx, (chunk, score) in enumerate(top_chunks):
        print(f"Rank {idx+1}, Score: {score:.4f}")
        print(chunk[:200] + "...\n")  # Display first 200 characters of the chunk
    
    # Text Processing using Async Programming
    print("Processing retrieved chunks asynchronously...")
    retrieved_texts = [chunk for chunk, _ in top_chunks]
    processed_chunks = asyncio.run(process_chunks(retrieved_texts))
    for idx, processed in enumerate(processed_chunks):
        print(f"Processed Chunk {idx+1}:")
        print(processed[:200] + "...\n")  # Display first 200 characters of processed text

if __name__ == "__main__":
    main()
