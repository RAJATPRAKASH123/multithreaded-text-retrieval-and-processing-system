import requests
from bs4 import BeautifulSoup
import re

class DataExtractor:
    def __init__(self, url):
        self.url = url

    def fetch_content(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception("Failed to fetch the URL content")

    def clean_text(self, html):
        soup = BeautifulSoup(html, "html.parser")
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator="\n")
        # Remove references like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        # Split into lines and remove extra spaces
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        # Stop processing after reaching sections like References or External links
        cleaned_text = []
        stop_sections = ['References', 'External links', 'See also']
        for line in lines:
            if any(stop in line for stop in stop_sections):
                break
            cleaned_text.append(line)
        return "\n".join(cleaned_text)

    def split_into_chunks(self, text, chunk_size=500):
        # Split text into chunks of approximately chunk_size words
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
        return chunks

    def extract_and_process(self):
        html = self.fetch_content()
        cleaned_text = self.clean_text(html)
        chunks = self.split_into_chunks(cleaned_text)
        return chunks
