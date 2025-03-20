#!/bin/bash

echo "Setting up the environment..."

# Install dependencies
# Upgrade pip using the correct command
python3 -m pip install --upgrade pip

# pip3 install --upgrade pip3
python3 -m pip install -r requirements.txt

# Install missing dependencies
# pip3 install sentence-transformers

# install bm25
pip3 install rank-bm25

# Download necessary NLTK data (tokenizer and stopwords)
python3 -m nltk.downloader punkt stopwords

# # Run the main script and log the output
python3 main.py | tee logs/output.log


# Run the test suite with pytest as a module
# PYTHONPATH=. python3 -m pytest --maxfail=1 --disable-warnings -v
# python3 -m pytest --maxfail=1 -v # --disable-warnings

echo "Setup complete. Check pipeline.log for details."
