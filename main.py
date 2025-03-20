# main.py (Updated to a PipelineManager for multiple URLs)

import os
import asyncio
import aiohttp
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.extraction import DataExtractor
from src.retrieval import TextRetriever
from src.logger import Logger
from src.cache_manager import CacheManager
from src.text_processing import TextProcessor  # Import Async Text Processor

class CustomException(Exception):
    """Custom exception for pipeline errors."""
    def __init__(self, message):
        super().__init__(message)
        Logger("exceptions.log", verbose=True).error(f"EXCEPTION: {message}")

class PipelineManager:
    def __init__(self, urls, query, max_words=300, chunk_strategy="context",
                 similarity_methods=None, normalization_methods=None, cache_file="cache.json"):
        """
        Parameters:
            urls (list of str): List of Wikipedia URLs to process.
            query (str): Query for retrieval.
            max_words (int): Maximum words per chunk.
            chunk_strategy (str): Chunking strategy to use (e.g., 'fixed', 'sliding', 'context', 'sentence').
            similarity_methods (list): Similarity methods to test (e.g., ["bm25", "tfidf", "word"]).
            normalization_methods (list): Normalization methods (e.g., ["min-max", "soft", "log"]).
            cache_file (str): Path to JSON file for caching processed URL results.

        """
        self.urls = urls
        self.query = query
        self.max_words = max_words
        self.chunk_strategy = chunk_strategy
        self.similarity_methods = similarity_methods or ["bm25", "tfidf", "word"]
        self.normalization_methods = normalization_methods or ["min-max", "soft", "log"]
        
        self.all_results_summary = []   # For combined plotting across URLs
        self.all_detailed_results = []  # Detailed output (per-URL) with chunk scores
        
        self.logger = Logger("pipeline.log", verbose=False)
        self.cache_manager = CacheManager(cache_file=cache_file)
        self.text_processor = TextProcessor()  # Initialize Text Processor

        # Ensure directories exist
        os.makedirs("plots", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("db", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    async def fetch_content_async(self, session, url):
        """Asynchronously fetch HTML content with logging and error handling."""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    Logger("network.log").log(f"HTTP error: {url} -> Status {response.status}")
                    return None
        except aiohttp.ClientError as e:
            self.logger.error(f"Request error for {url}: {e}")
            return None
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout fetching {url}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching {url}: {e}")
            return None


    async def process_url(self, url):
        """
        Processes a single URL: 
          1) Extract & chunk text 
          2) Filter chunks 
          3) Run retrieval experiments for each (similarity_method, normalization_method)
        Returns a dictionary with results summary & detailed info for this URL.

        Uses caching to avoid duplicate processing.
        """
        # Check cache first.
        cached_result = await self.cache_manager.get(url)
        if cached_result:
            self.logger.log(f"Cache hit for {url}. Using cached results.")
            return cached_result


        # 1) Fetch HTML content (synchronously if you prefer, or re-use existing code)
        #    We'll do a single fetch here using requests for simplicity 
        #    (or you can keep the async approach from fetch_content_async).
        
        # If you want to do it fully async, you can reuse fetch_content_async instead:
        async with aiohttp.ClientSession() as session:
            html_content = await self.fetch_content_async(session, url)
        if not html_content:
            raise CustomException(f"Failed to fetch content for URL: {url}")
            
        
        # response = requests.get(url)
        # if response.status_code != 200:
        #     print(f"Failed to fetch {url}. Status code: {response.status_code}")
        #     return None
        # html_content = response.text

        # 2) Extract & chunk text
        extractor = DataExtractor(max_words=self.max_words, chunk_strategy=self.chunk_strategy)
        extracted_text = extractor.extract_text(html_content)
        chunks = extractor.create_chunks(extracted_text)
        
        if not chunks:
            raise CustomException(f"No content extracted from {url}")
            return None
        
        # Use BM25-based dummy retrieval to filter out unimportant chunks
        dummy_retriever = TextRetriever(method="bm25")
        filtered_chunks = dummy_retriever.similarity_calculator.filter_chunks(chunks)

        if not filtered_chunks:
            self.logger.log(f"[{url}] No relevant chunks found.")
            return None
        
        self.logger.log(f"[{url}] Extracted {len(filtered_chunks)} filtered chunks.")


        # 3) For each combination of (similarity_method, normalization_method), retrieve top chunks
        url_results_summary = []
        url_detailed_results = []

        for sim_method in self.similarity_methods:
            for norm_method in self.normalization_methods:
                retriever = TextRetriever(method=sim_method)
                top_chunks = retriever.retrieve_top_chunks(
                    query=self.query,
                    chunks=filtered_chunks,
                    top_k=3,
                    normalization_method=norm_method
                )
                if top_chunks:
                    avg_score = sum(score for _, score in top_chunks) / len(top_chunks)
                else:
                    avg_score = 0                
                
                label = f"{sim_method} | {norm_method}"
                
                processed_chunks = await self.text_processor.process_chunks([chunk for chunk, _ in top_chunks])

                # Store detailed results
                url_detailed_results.append({
                    "URL": url,
                    "Similarity Method": sim_method,
                    "Normalization Method": norm_method,
                    "Average Score": avg_score,
                    "Top Chunks": list(zip(processed_chunks, [score for _, score in top_chunks]))
                })
                # Store summary results
                url_results_summary.append({
                    "URL": url,
                    "Similarity Method": sim_method,
                    "Normalization Method": norm_method,
                    "Average Score": avg_score
                })
                self.logger.log(f"[{url}] {label} -> Avg Score: {avg_score:.4f}")
                # print(f"[{url}] Combination: {label} -> Average Score: {avg_score:.4f}")

        result = {
            "url_results_summary": url_results_summary,
            "url_detailed_results": url_detailed_results
        }
        await self.cache_manager.set(url, result)  # Await it
        return result

    async def run_all_urls(self):
        """Runs the pipeline for all URLs concurrently."""
        tasks = []
        for url in self.urls:
            tasks.append(self.process_url(url))
        
        results = await asyncio.gather(*tasks)
        # Filter out None
        # Handle errors properly without stopping execution
        valid_results = []
        for res in results:
            if isinstance(res, Exception):
                self.logger.error(f"Unhandled exception in task: {res}")
            elif res is not None:
                valid_results.append(res)

        # Combine summary and detailed results across all URLs
        for item in valid_results:
            self.all_results_summary.extend(item["url_results_summary"])
            self.all_detailed_results.extend(item["url_detailed_results"])
        return valid_results

    def plot_results(self):
        """Creates plots (heatmap, bar chart) for the combined results across all URLs."""
        if not self.all_results_summary:
            print("No results to plot.")
            return
        
        df = pd.DataFrame(self.all_results_summary)
        
        # For plotting, pivot by (Similarity Method, Normalization Method), ignoring URLs
        pivot_table = df.pivot_table(
            index="Similarity Method",
            columns="Normalization Method",
            values="Average Score",
            aggfunc="mean"
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
        plt.title("Heatmap of Average Normalized Scores (All URLs)")
        plt.savefig("plots/heatmap_scores.png")
        plt.show()

        # Grouped bar plot, ignoring the URL dimension
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df,
            x="Similarity Method",
            y="Average Score",
            hue="Normalization Method",
            palette="Set2"
        )
        plt.title("Average Normalized Scores by Similarity & Normalization Methods (All URLs)")
        plt.ylim(0, 1)
        plt.xlabel("Similarity Method")
        plt.ylabel("Average Normalized Score")
        plt.savefig("plots/grouped_bar_scores.png")
        plt.show()

    def save_results(self):
        """Saves the combined detailed results for all URLs."""
        if not self.all_detailed_results:
            print("No detailed results to save.")
            return
        
        results_folder = "results"
        os.makedirs(results_folder, exist_ok=True)
        with open(os.path.join(results_folder, "results.txt"), "w", encoding="utf-8") as f:
            for res in self.all_detailed_results:
                f.write(f"URL: {res['URL']}\n")
                f.write(f"Similarity Method: {res['Similarity Method']}, Normalization Method: {res['Normalization Method']}\n")
                f.write(f"Average Score: {res['Average Score']:.4f}\n")
                f.write("Top Chunks:\n")
                for idx, (chunk, score) in enumerate(res["Top Chunks"], start=1):
                    f.write(f"  Rank {idx}: Score: {score:.4f}\n")
                    f.write(f"  Chunk: {chunk[:200]}...\n\n")  # Truncate chunk display
                f.write("=" * 50 + "\n\n")

    def run_pipeline(self):
        """Main entry point to run the pipeline for all URLs."""
        # 1) Async run all URLs
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self.run_all_urls())
        if not results:
            self.logger.log("No valid results processed.")
            return

        # 2) Plot results
        self.plot_results()

        # 3) Save results
        self.save_results()

        self.logger.log("Pipeline completed successfully.")

if __name__ == "__main__":
    # Example usage with multiple Wikipedia URLs
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Deep_learning"
    ]
    query = "What is the impact of AI on society?"
    manager = PipelineManager(urls, query, max_words=300, chunk_strategy="context")
    manager.run_pipeline()
