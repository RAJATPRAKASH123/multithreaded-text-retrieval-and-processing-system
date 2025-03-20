import os
import json

class CacheManager:
    """
    Manages caching of processed URL results to avoid redundant processing.
    Cache is stored in a JSON file.
    """
    def __init__(self, cache_file="cache.json"):
        self.cache_file = cache_file
        self.cache = self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
                return {}
        return {}

    def save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def get(self, key):
        """Returns the cached value for the given key, or None if not present."""
        return self.cache.get(key, None)

    def set(self, key, value):
        """Sets the cache value for the given key and saves the cache."""
        self.cache[key] = value
        self.save_cache()
