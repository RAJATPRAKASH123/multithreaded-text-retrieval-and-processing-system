import os
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.extraction import DataExtractor
from src.retrieval import TextRetriever

# Define similarity and normalization methods to test
SIMILARITY_METHODS = ["bm25", "tfidf", "word"]
NORMALIZATION_METHODS = ["min-max", "soft", "log"]

# Wikipedia URL
wiki_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"

# Initialize the extractor with text cleaning
extractor = DataExtractor(max_words=300, chunk_strategy="context_aware")

# Fetch and extract Wikipedia text
response = requests.get(wiki_url)
html_content = response.text if response.status_code == 200 else ""
extracted_text = extractor.extract_text(html_content)
chunks = extractor.create_chunks(extracted_text)

# Use a dummy retriever instance to filter out unimportant chunks
dummy_retriever = TextRetriever(method="bm25")
filtered_chunks = dummy_retriever.similarity_calculator.filter_chunks(chunks)
print(f"Extracted {len(filtered_chunks)} filtered chunks.")

# query = "What is the impact of AI on society?"
query = "AI good or bad?"

# Store results for each combination in a list of dictionaries, including detailed top chunks info
detailed_results = []
results_summary = []  # For plotting summary

for sim_method in SIMILARITY_METHODS:
    for norm_method in NORMALIZATION_METHODS:
        # Initialize a retriever with the selected similarity method
        retriever = TextRetriever(method=sim_method)
        # Retrieve top 3 chunks using the current normalization method
        top_chunks = retriever.retrieve_top_chunks(query, filtered_chunks, top_k=3, normalization_method=norm_method)
        # Calculate the average normalized score over the top retrieved chunks
        if top_chunks:
            avg_score = sum(score for chunk, score in top_chunks) / len(top_chunks)
        else:
            avg_score = 0
        label = f"{sim_method} | {norm_method}"
        detailed_results.append({
            "Similarity Method": sim_method,
            "Normalization Method": norm_method,
            "Average Score": avg_score,
            "Top Chunks": top_chunks
        })
        results_summary.append({
            "Similarity Method": sim_method,
            "Normalization Method": norm_method,
            "Average Score": avg_score
        })
        print(f"Combination: {label} -> Average Score: {avg_score:.4f}")

# Convert summary results to a DataFrame for plotting
df = pd.DataFrame(results_summary)

# Create a heatmap of average normalized scores
pivot_table = df.pivot("Similarity Method", "Normalization Method", "Average Score")
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
plt.title("Heatmap of Average Normalized Scores")
# Save the heatmap plot in the plots folder
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/heatmap_scores.png")
plt.show()

# Create a grouped bar plot for a more detailed comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Similarity Method", y="Average Score", hue="Normalization Method", palette="Set2")
plt.title("Average Normalized Scores by Similarity & Normalization Methods")
plt.ylim(0, 1)
plt.xlabel("Similarity Method")
plt.ylabel("Average Normalized Score")
plt.savefig("plots/grouped_bar_scores.png")
plt.show()

# Write detailed results to a text file in the results folder
results_folder = "results"
os.makedirs(results_folder, exist_ok=True)
with open(os.path.join(results_folder, "results.txt"), "w", encoding="utf-8") as f:
    for res in detailed_results:
        f.write(f"Similarity Method: {res['Similarity Method']}, Normalization Method: {res['Normalization Method']}\n")
        f.write(f"Average Score: {res['Average Score']:.4f}\n")
        f.write("Top Chunks:\n")
        for idx, (chunk, score) in enumerate(res["Top Chunks"], start=1):
            f.write(f"  Rank {idx}: Score: {score:.4f}\n")
            f.write(f"  Chunk: {chunk}\n\n")
        f.write("=" * 50 + "\n\n")
