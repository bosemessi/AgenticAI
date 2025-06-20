#!/bin/sh
export PYTHONPATH=.
source .venv/Scripts/activate
# This script runs the RAG files in the specified directory.
# First python script
python -m scripts.001_simple_RAG --txt_file_path=data/cat-facts.txt
# Second python script
