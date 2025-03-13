#!/bin/bash

# update pip3
python3 -m pip3 install --upgrade pip3

# Install dependencies
pip3 install -r requirements.txt

# Download necessary NLTK data (tokenizer and stopwords)
python3 -m nltk.downloader punkt stopwords

# Run the main program and log the output
python3 main.py | tee output.log
