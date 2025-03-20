# Multi-Threaded Text Retrieval and Processing System

## Overview
This project implements a **class-based, multi-threaded** system that simulates a simplified Retrieval-Augmented Generation (RAG) pipeline. It:

1. **Scrapes Wikipedia** (e.g., an article on Artificial Intelligence).
2. **Cleans and splits** the text into chunks using multiple strategies (fixed, sentence-based, context-based, etc.).
3. **Retrieves relevant chunks** using different similarity methods (BM25, TF-IDF, word embeddings).
4. **Normalizes** similarity scores with min-max, soft, or log scaling.
5. **Plots** the results (heatmaps, bar charts) for easy comparison.
6. **Logs** detailed output in a text file and stores retrieval scores in an SQLite database.

---

## Key Features
- **Class-Based Pipeline**  
  A `RetrievalPipeline` (or `AIPipeline`) encapsulates the workflow: extraction, retrieval experiments, plotting, and saving results.
- **Multiple Chunking Strategies**  
  Choose from **fixed-size**, **context-aware**, **sentence-based**, or **sliding-window** approaches.
- **Multiple Similarity & Normalization Methods**  
  - Similarity: **BM25**, **TF-IDF**, or **Word Embeddings (GloVe)**  
  - Normalization: **min-max**, **soft**, or **log**  
- **Multi-Threaded Retrieval**  
  Uses **threading** to compute similarity scores more efficiently.  
- **Database & Logs**  
  - Stores chunk scores in an **SQLite** database (`db/retrieval_results.db`).  
  - Outputs logs to `logs/output.log`, plots to `plots/`, and detailed chunk retrieval results to `results/`.

---

## Folder Structure

```
MULTITHREADED-TEXT-RETRIEVAL-AND-PROCESSING-SYSTEM/
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── setup.sh                 # Installs dependencies, runs the pipeline
├── main.py                  # Example entry point or pipeline runner
├── data/                    # (Optional) Raw data files or samples
├── db/
│   └── retrieval_results.db # SQLite database storing retrieval scores
├── logs/
│   ├── output.log           # Log of program output
│   ├── heatmap_scores.png   # (Optional) If you save plots here
│   └── grouped_bar_scores.png
├── plots/
│   ├── heatmap_scores.png   # Heatmap of average normalized scores
│   └── grouped_bar_scores.png
├── results/
│   └── results.txt          # Detailed chunk retrieval results
└── src/
    ├── __init__.py          # Marks src as a Python package
    ├── extraction.py        # DataExtractor for scraping & cleaning
    ├── embedding.py         # (Optional) For specialized embedding logic
    ├── processing.py        # (Optional) For async text processing
    ├── retrieval.py         # TextRetriever for multi-threaded retrieval & DB storage
    ├── similarity.py        # (Optional) If you keep a separate similarity module
    └── text_similarity.py   # TextSimilarity class (BM25, TF-IDF, word embeddings)
```

---

## Installation & Setup

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/YourUsername/YourRepoName.git
   cd YourRepoName
   ```

2. **Install Dependencies**  
   Ensure Python 3.7+ is installed, then run:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
   This installs Python packages from `requirements.txt` and downloads necessary NLTK data.

3. **Run the Pipeline**  
   ```bash
   python main.py
   ```
   or if you’ve placed the pipeline logic in another file, adjust the command accordingly. The program logs output to `logs/output.log`, saves plots to `plots/`, and writes a detailed text file in `results/`.

---

## Usage

### **Class-Based Pipeline**
A typical usage example (in `main.py`):
```python
if __name__ == "__main__":
    wiki_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    query = "What is the impact of AI on society?"
    pipeline = RetrievalPipeline(
        wiki_url=wiki_url,
        query=query,
        max_words=300,
        chunk_strategy="context",  # Could also be "fixed", "sentence", "sliding"
        similarity_methods=["bm25", "tfidf", "word"],
        normalization_methods=["min-max", "soft", "log"]
    )
    pipeline.run_pipeline()
```
- **Chunk Strategy**: Choose `"fixed"`, `"sentence"`, `"sliding"`, etc.  
- **Similarity Methods**: `["bm25", "tfidf", "word"]`  
- **Normalization Methods**: `["min-max", "soft", "log"]`

### **Generated Outputs**
- **Plots** (`plots/`): Heatmaps, grouped bar charts comparing average normalized scores.  
- **results.txt** (`results/`): Detailed top chunk info (rank, score, chunk text).  
- **SQLite DB** (`db/retrieval_results.db`): Stores retrieval scores.  

---

## Extending the System

1. **Add More Chunking Strategies**  
   - Implement custom chunking logic in `extraction.py` (e.g., topic-based or advanced paragraph grouping).
2. **Incorporate New Similarity Methods**  
   - Modify `text_similarity.py` to add new algorithms or embeddings.
3. **Scale Up**  
   - For large corpora, integrate **FAISS** or another vector database to handle retrieval more efficiently.
4. **Async Processing**  
   - Add asynchronous text processing in `processing.py` for large-scale or concurrent tasks.

---

## Troubleshooting
 
- **NLTK Runtime Warnings**: Common if `nltk.downloader` is imported in an unexpected order. Usually harmless.  
- **Plots Not Showing**: Make sure `matplotlib` and `seaborn` are installed and the script is running in an environment that supports GUI or inline plotting.  
- **Slow Performance**: Increase thread pool size, or reduce chunk size if you have a very large text.

---

## License
This project is released under the [MIT License](LICENSE). You are free to modify and distribute the code as permitted by the license terms.

---

## Acknowledgments
- **Python** for the core language.
- **Requests** + **BeautifulSoup** for web scraping.
- **gensim** for word embeddings (GloVe).
- **rank-bm25**, **scikit-learn** for BM25/TF-IDF.
- **SQLite** for results storage.
- **Seaborn** + **matplotlib** for plotting.

Feel free to **contribute** by opening issues or pull requests. Happy experimenting!

