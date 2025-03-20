import asyncio
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from src.logger import Logger

class TextProcessor:
    """Asynchronous text processing for tokenization and stopword removal."""

    def __init__(self):
        self.logger = Logger("text_processing.log")
        nltk.download("punkt")
        nltk.download("stopwords")
        self.stop_words = set(stopwords.words("english"))

    async def process_chunk(self, chunk):
        """Tokenizes and removes stopwords asynchronously for a single chunk."""
        try:
            words = word_tokenize(chunk)
            processed_words = [word.lower() for word in words if word.isalnum() and word.lower() not in self.stop_words]
            return " ".join(processed_words)
        except Exception as e:
            self.logger.error(f"Error processing chunk: {e}")
            return chunk  # Return original chunk in case of failure

    async def process_chunks(self, chunks):
        """Processes multiple chunks concurrently."""
        tasks = [self.process_chunk(chunk) for chunk in chunks]
        return await asyncio.gather(*tasks)
