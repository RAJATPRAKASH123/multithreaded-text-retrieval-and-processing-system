import asyncio
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

async def preprocess_chunk(chunk):
    # Get the event loop and run tokenization in an executor
    loop = asyncio.get_event_loop()
    tokens = await loop.run_in_executor(None, word_tokenize, chunk)
    stops = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stops]
    return " ".join(filtered_tokens)

async def process_chunks(chunks):
    tasks = [preprocess_chunk(chunk) for chunk in chunks]
    return await asyncio.gather(*tasks)
