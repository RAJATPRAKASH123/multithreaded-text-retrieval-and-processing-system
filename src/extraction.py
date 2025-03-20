import re
import time
import logging
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from src.logger import Logger

class DataExtractor:
    """Extracts and processes text from Wikipedia HTML. Feel free to set chunk_strategy to 'fixed', 'sentence', 'sliding' and 'context'."""
    def __init__(self, max_words=300, chunk_strategy="context"):
        self.max_words = max_words
        self.chunk_strategy = chunk_strategy
        self.logger = Logger()
        
## depreciated function for fetching content[requests.get()]

    # def fetch_content(self, url, retries=3, delay=5, timeout=10):
    #     """Fetches Wikipedia page content with retries and logging."""
    #     for attempt in range(1, retries + 1):
    #         try:
    #             response = requests.get(url, timeout=timeout)
    #             response.raise_for_status()
    #             if response.status_code == 200:
    #                 return response.text
            
    #         except requests.exceptions.Timeout:
    #             self.logger.error(f"Timeout error on attempt {attempt} for URL: {url}")
    #             time.sleep(delay * attempt)  # Exponential backoff
            
    #         except requests.exceptions.ConnectionError:
    #             self.logger.error(f"Connection error on attempt {attempt} for URL: {url}")
    #             time.sleep(delay * attempt)
            
    #         except requests.exceptions.HTTPError as e:
    #             self.logger.error(f"HTTP error {response.status_code} on attempt {attempt} for URL: {url}: {e}")
    #             time.sleep(delay * attempt)
            
    #         except requests.exceptions.RequestException as e:
    #             self.logger.error(f"Request error on attempt {attempt} for URL: {url}: {e}")
    #             time.sleep(delay * attempt)

    #     self.logger.error(f"Failed to fetch content after {retries} retries for URL: {url}")
    #     return None  # Return None instead of raising an exception

    def extract_text(self, html):
        """Extracts clean text from Wikipedia content."""
        soup = BeautifulSoup(html, "html.parser")
        content_div = soup.find("div", class_="mw-body-content")
        if not content_div:
            raise Exception("Content division not found!")
        
        # Remove navigation, footers, sidebars, citations, etc.
        for element in content_div.find_all(["nav", "footer", "aside", "sup", "cite"]):
            element.decompose()
        
        # Extract paragraphs
        paragraphs = [p.get_text() for p in content_div.find_all("p") if p.get_text().strip()]
        
        # Extract list items (filtering out generic ones)
        list_items = [
            li.get_text() for ul in content_div.find_all("ul")
            if "References" not in ul.get_text() and "See also" not in ul.get_text()
            for li in ul.find_all("li") if li.get_text().strip()
        ]
        
        # Extract table data from <td> elements
        table_cells = [
            td.get_text() for table in content_div.find_all("table")
            for tr in table.find_all("tr")
            for td in tr.find_all("td") if td.get_text().strip()
        ]
        
        # Extract headings (filtering out non-relevant ones)
        headings = [
            h.get_text() for h in content_div.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
            if "See also" not in h.get_text() and "References" not in h.get_text()
        ]
        
        all_text = paragraphs + list_items + table_cells + headings
        return [self.clean_text(text) for text in all_text]

    def clean_text(self, text):
        """Removes Wikipedia-specific elements (references, headers, URLs, extra spaces)."""
        text = re.sub(r"\[\d+\]", "", text)         # Remove references like [1], [23]
        text = re.sub(r"==.*?==", "", text)           # Remove section headers
        text = re.sub(r"https?://\S+", "", text)      # Remove URLs
        text = re.sub(r"\s+", " ", text).strip()       # Remove extra spaces
        return text

    def create_chunks(self, extracted_text, strategy=None, overlap=50):
        """
        Splits text into chunks based on the selected strategy, handling empty chunks.
        
        Parameters:
            extracted_text (list): List of cleaned text segments.
            strategy (str): 'fixed', 'sliding', 'context', or 'sentence'.
                          Defaults to the instance's chunk_strategy.
            overlap (int): Number of words to overlap for sliding window strategy.
        
        Returns:
            List of text chunks.
        """
        strategy = strategy or self.chunk_strategy
        if not extracted_text or all(not text.strip() for text in extracted_text):
            print("Warning: No valid text found for chunking.")
            return []
        
        chunks = []
        if strategy == "fixed":
            current_chunk, word_count = [], 0
            for text in extracted_text:
                words = text.split()
                if word_count + len(words) > self.max_words and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk, word_count = [], 0
                if words:
                    current_chunk.append(text)
                    word_count += len(words)
            if current_chunk:
                chunks.append(" ".join(current_chunk))
        
        elif strategy == "sliding":
            words = " ".join(extracted_text).split()
            if not words:
                print("Warning: No valid words found for sliding chunking.")
                return []
            step = self.max_words - overlap
            for i in range(0, len(words), step):
                chunk = " ".join(words[i:i+self.max_words])
                if chunk.strip():
                    chunks.append(chunk)
        
        elif strategy == "context":
            current_chunk, word_count = [], 0
            for text in extracted_text:
                words = text.split()
                if word_count + len(words) > self.max_words and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk, word_count = [text], len(words)
                else:
                    if words:
                        current_chunk.append(text)
                        word_count += len(words)
            if current_chunk:
                chunks.append(" ".join(current_chunk))
        
        elif strategy == "sentence":
            # from nltk.tokenize import sent_tokenize
            sentences = [sent for text in extracted_text for sent in sent_tokenize(text) if sent.strip()]
            if not sentences:
                print("Warning: No valid sentences found for sentence-based chunking.")
                return []
            current_chunk, word_count = [], 0
            for sentence in sentences:
                words = sentence.split()
                if word_count + len(words) > self.max_words and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk, word_count = [sentence], len(words)
                else:
                    current_chunk.append(sentence)
                    word_count += len(words)
            if current_chunk:
                chunks.append(" ".join(current_chunk))
        else:
            raise ValueError("Invalid chunking strategy. Choose 'fixed', 'sliding', 'context', or 'sentence'.")
        
        if not chunks:
            print("Warning: No chunks were created after processing.")
        return chunks
