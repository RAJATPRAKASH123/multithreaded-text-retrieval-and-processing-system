# Multi-Threaded Text Retrieval and Processing System

## Overview
This project implements a lightweight, multi-threaded system that simulates a simplified Retrieval-Augmented Generation (RAG) pipeline. It performs the following:
- **Data Extraction and Cleaning:** Scrapes the Wikipedia page on Artificial Intelligence, cleans the HTML content, removes references, and splits the text into manageable chunks.
- **Embedding Creation:** Uses Python’s multiprocessing module to compute an embedding for each text chunk. Here, we use a simple average of pre-trained word vectors (via the GloVe model from gensim).
- **Document Retrieval:** Uses threading to compute cosine similarity between a query embedding and each chunk’s embedding, retrieving the top three most relevant chunks.
- **Text Processing:** Uses asynchronous programming (asyncio) to concurrently preprocess the retrieved chunks (tokenization and stopword removal).
- **Linux Integration:** A Bash script (`setup.sh`) is provided to install dependencies, download necessary NLTK data, run the system, and log the output.

## File Structure
- **README.md:** This file.
- **requirements.txt:** Lists the Python dependencies.
- **setup.sh:** Bash script to set up the environment and run the program.
- **main.py:** Main driver that ties all modules together.
- **extraction.py:** Contains the `DataExtractor` class for scraping and cleaning the Wikipedia page.
- **embedding.py:** Contains the `EmbeddingCreator` class that computes embeddings using multiprocessing.
- **retrieval.py:** Contains the `DocumentRetriever` class that retrieves top relevant chunks using threading.
- **processing.py:** Contains asynchronous functions to process retrieved text.

## Setup Instructions
1. Ensure you have Python 3.7+ installed.
2. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
