#!/bin/bash

echo "Setting up the environment..."

# Install dependencies
pip3 install --upgrade pip3
pip3 install -r requirements.txt

# Install missing dependencies
# pip3 install sentence-transformers

# install bm25
pip3 install rank-bm25

# Download necessary NLTK data (tokenizer and stopwords)
python3 -m nltk.downloader punkt stopwords

# Run the main script and log the output
python3 main.py | tee logs/output.log

echo "Setup complete. Check output.log for details."
