import re
from bs4 import BeautifulSoup

class DataExtractor:
    """Extracts and processes text from Wikipedia HTML."""

    def __init__(self, max_words=300, chunk_strategy="context_aware"):
        self.max_words = max_words
        self.chunk_strategy = chunk_strategy

    def fetch_content(self, url):
        """Fetches Wikipedia page content."""
        import requests
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Failed to retrieve page. Status code: {response.status_code}")

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

    def create_chunks(self, extracted_text):
        """Splits text into chunks for retrieval."""
        chunks, current_chunk = [], []
        word_count = 0

        for text in extracted_text:
            words = text.split()
            if word_count + len(words) > self.max_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                word_count = 0
            current_chunk.append(text)
            word_count += len(words)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
