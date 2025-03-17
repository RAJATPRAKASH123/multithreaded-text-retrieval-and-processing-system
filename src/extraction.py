import re
import time
from bs4 import BeautifulSoup
import logging

class DataExtractor:
    """Extracts and processes text from Wikipedia HTML."""

    def __init__(self, max_words=300, chunk_strategy="context"):
        self.max_words = max_words
        self.chunk_strategy = chunk_strategy

    # Set up logging
    logging.basicConfig(filename="logs/error_log.txt", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

    def fetch_content(self, url, retries=3, delay=5, timeout=10):
        """Fetches Wikipedia page content with retries, error handling, and logging."""
        for attempt in range(1, retries + 1):
            try:
                response = requests.get(url, timeout=timeout)  # Set timeout
                response.raise_for_status()  # Raise an exception for HTTP errors

                if response.status_code == 200:
                    return response.text
            
            except requests.exceptions.Timeout:
                logging.error(f"Timeout error on attempt {attempt} for URL: {url}")
                print(f"Timeout on attempt {attempt}. Retrying in {delay} seconds...")
            
            except requests.exceptions.ConnectionError:
                logging.error(f"Connection error on attempt {attempt} for URL: {url}")
                print(f"Connection error on attempt {attempt}. Retrying in {delay} seconds...")
            
            except requests.exceptions.HTTPError as e:
                logging.error(f"HTTP error {response.status_code} on attempt {attempt} for URL: {url}: {e}")
                print(f"HTTP error {response.status_code} on attempt {attempt}. Retrying in {delay} seconds...")
            
            except requests.exceptions.RequestException as e:
                logging.error(f"General request error on attempt {attempt} for URL: {url}: {e}")
                print(f"Request error on attempt {attempt}. Retrying in {delay} seconds...")

            time.sleep(delay * attempt)  # Exponential backoff (waits longer after each failure)

        logging.error(f"Failed to fetch content after {retries} retries for URL: {url}")
        raise Exception(f"Failed to retrieve page after {retries} retries.")

    def extract_text(self, html):
        """Extracts clean text from Wikipedia content."""
        soup = BeautifulSoup(html, "html.parser")
        content_div = soup.find("div", class_="mw-body-content")
        if not content_div:
            raise Exception("Content division not found!")

        # Remove Wikipedia navigation, references, and sidebars
        for element in content_div.find_all(["nav", "footer", "aside", "sup", "cite"]):
            element.decompose()

        # Extract paragraphs
        paragraphs = [p.get_text() for p in content_div.find_all("p") if p.get_text().strip()]

        # Extract list items (only if they don't belong to navigation)
        list_items = [
            li.get_text() for ul in content_div.find_all("ul") 
            if "References" not in ul.get_text() and "See also" not in ul.get_text()
            for li in ul.find_all("li") if li.get_text().strip()
        ]

        # Extract table data
        table_cells = [td.get_text() for table in content_div.find_all("table") for tr in table.find_all("tr") for td in tr.find_all("td") if td.get_text().strip()]

        # Extract headings (only if relevant)
        headings = [
            h.get_text() for h in content_div.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]) 
            if "See also" not in h.get_text() and "References" not in h.get_text()
        ]

        all_text = paragraphs + list_items + table_cells + headings
        return [self.clean_text(text) for text in all_text]  # Apply cleaning


    def clean_text(self, text):
        """Removes Wikipedia-specific elements (footers, links, metadata)."""
        text = re.sub(r"\[\d+\]", "", text)  # Remove references like [1], [23]
        text = re.sub(r"==.*?==", "", text)  # Remove Wikipedia section headers
        text = re.sub(r"https?://\S+", "", text)  # Remove URLs
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
        return text
        
    def create_chunks(self, extracted_text, strategy="context", overlap=50):
        """Splits text into chunks based on the selected strategy, handling empty chunks.

        Parameters:
            extracted_text (list): List of cleaned text segments.
            strategy (str): 'fixed', 'sliding', 'context', 'sentence'
            overlap (int): Number of words to overlap for sliding window strategy.

        Returns:
            List of text chunks.
        """
        if not extracted_text or all(not text.strip() for text in extracted_text):
            print("Warning: No valid text found for chunking.")
            return []

        chunks = []
        
        if strategy == "fixed":
            # Fixed-size chunking
            current_chunk, word_count = [], 0
            for text in extracted_text:
                words = text.split()
                if word_count + len(words) > self.max_words and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk, word_count = [], 0
                if words:  # Ensure text is not empty
                    current_chunk.append(text)
                    word_count += len(words)
            if current_chunk:
                chunks.append(" ".join(current_chunk))

        elif strategy == "sliding":
            # Overlapping sliding window chunking
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
            # Context-aware chunking
            current_chunk, word_count = [], 0
            for text in extracted_text:
                words = text.split()
                if word_count + len(words) > self.max_words and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk, word_count = [text], len(words)
                else:
                    if words:  # Ensure text is not empty
                        current_chunk.append(text)
                        word_count += len(words)
            if current_chunk:
                chunks.append(" ".join(current_chunk))

        elif strategy == "sentence":
            # Sentence-based chunking
            from nltk.tokenize import sent_tokenize
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
