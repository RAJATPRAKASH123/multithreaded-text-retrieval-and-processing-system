import json
import asyncio
from src.logger import Logger

class CacheManager:
    """Handles caching of processed URLs to avoid redundant computation."""
    
    def __init__(self, cache_file="cache.json"):
        self.cache_file = cache_file
        self.logger = Logger("cache.log")
        self.lock = asyncio.Lock()  # 🔹 Ensures thread-safe writes

    async def get(self, url):
        """Fetch cached data asynchronously."""
        async with self.lock:
            try:
                with open(self.cache_file, "r") as f:
                    cache_data = json.load(f)
                return cache_data.get(url, None)
            except (FileNotFoundError, json.JSONDecodeError):
                return None

    async def set(self, url, data):
        """Save results to cache asynchronously."""
        async with self.lock:
            try:
                cache_data = {}
                try:
                    with open(self.cache_file, "r") as f:
                        cache_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    pass

                cache_data[url] = data
                with open(self.cache_file, "w") as f:
                    json.dump(cache_data, f, indent=4)
                
                self.logger.log(f"Cached results for {url}.")
            except Exception as e:
                self.logger.error(f"Error writing cache: {e}")
