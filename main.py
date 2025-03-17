import os
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.extraction import DataExtractor
from src.retrieval import TextRetriever

class RetrievalPipeline:
    def __init__(self, wiki_url, query, max_words=300, chunk_strategy="context_aware",
                 similarity_methods=None, normalization_methods=None):
        """
        Parameters:
            wiki_url (str): URL of the Wikipedia page.
            query (str): Query for retrieval.
            max_words (int): Maximum words per chunk.
            chunk_strategy (str): Chunking strategy to use.
            similarity_methods (list): List of similarity methods to test (e.g., ["bm25", "tfidf", "word"]).
            normalization_methods (list): List of normalization methods (e.g., ["min-max", "soft", "log"]).
        """
        self.wiki_url = wiki_url
        self.query = query
        self.max_words = max_words
        self.chunk_strategy = chunk_strategy
        self.similarity_methods = similarity_methods or ["bm25", "tfidf", "word"]
        self.normalization_methods = normalization_methods or ["min-max", "soft", "log"]
        self.results_summary = []  # For plotting
        self.detailed_results = []  # Detailed output including individual chunk scores
        self.chunks = None
        self.filtered_chunks = None

    def run_extraction(self):
        extractor = DataExtractor(max_words=self.max_words, chunk_strategy=self.chunk_strategy)
        response = requests.get(self.wiki_url)
        html_content = response.text if response.status_code == 200 else ""
        extracted_text = extractor.extract_text(html_content)
        self.chunks = extractor.create_chunks(extracted_text)
        # Use a dummy retriever (BM25) to filter out unimportant chunks
        dummy_retriever = TextRetriever(method="bm25")
        self.filtered_chunks = dummy_retriever.similarity_calculator.filter_chunks(self.chunks)
        print(f"Extracted {len(self.filtered_chunks)} filtered chunks.")

    def run_experiments(self):
        for sim_method in self.similarity_methods:
            for norm_method in self.normalization_methods:
                retriever = TextRetriever(method=sim_method)
                top_chunks = retriever.retrieve_top_chunks(self.query, self.filtered_chunks,
                                                             top_k=3, normalization_method=norm_method)
                if top_chunks:
                    avg_score = sum(score for _, score in top_chunks) / len(top_chunks)
                else:
                    avg_score = 0
                label = f"{sim_method} | {norm_method}"
                self.detailed_results.append({
                    "Similarity Method": sim_method,
                    "Normalization Method": norm_method,
                    "Average Score": avg_score,
                    "Top Chunks": top_chunks
                })
                self.results_summary.append({
                    "Similarity Method": sim_method,
                    "Normalization Method": norm_method,
                    "Average Score": avg_score
                })
                print(f"Combination: {label} -> Average Score: {avg_score:.4f}")

    def plot_results(self):
        df = pd.DataFrame(self.results_summary)
        # Create heatmap
        pivot_table = df.pivot("Similarity Method", "Normalization Method", "Average Score")
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
        plt.title("Heatmap of Average Normalized Scores")
        plt.savefig("plots/heatmap_scores.png")
        plt.show()

        # Create grouped bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Similarity Method", y="Average Score", hue="Normalization Method", palette="Set2")
        plt.title("Average Normalized Scores by Similarity & Normalization Methods")
        plt.ylim(0, 1)
        plt.xlabel("Similarity Method")
        plt.ylabel("Average Normalized Score")
        plt.savefig("plots/grouped_bar_scores.png")
        plt.show()

    def save_results(self):
        results_folder = "results"
        os.makedirs(results_folder, exist_ok=True)
        with open(os.path.join(results_folder, "results.txt"), "w", encoding="utf-8") as f:
            for res in self.detailed_results:
                f.write(f"Similarity Method: {res['Similarity Method']}, Normalization Method: {res['Normalization Method']}\n")
                f.write(f"Average Score: {res['Average Score']:.4f}\n")
                f.write("Top Chunks:\n")
                for idx, (chunk, score) in enumerate(res["Top Chunks"], start=1):
                    f.write(f"  Rank {idx}: Score: {score:.4f}\n")
                    f.write(f"  Chunk: {chunk}\n\n")
                f.write("=" * 50 + "\n\n")

    def run_pipeline(self):
        self.run_extraction()
        self.run_experiments()
        self.plot_results()
        self.save_results()

if __name__ == "__main__":
    wiki_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    query = "What is the impact of AI on society?"
    pipeline = RetrievalPipeline(wiki_url, query, max_words=300, chunk_strategy="context_aware")
    pipeline.run_pipeline()
