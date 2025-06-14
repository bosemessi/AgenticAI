import pymupdf
import os
import numpy as np
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('NEBIUS_API_KEY')

import sys
print(sys.path)

from src.datareader_utils.datareader import read_txt_files
from argparse import ArgumentParser

def arg_parser():
    parser = ArgumentParser(description="Run a simple RAG pipeline with OpenAI.")
    parser.add_argument("--txt_file_path", type=str, required=True, help="Path to the text file to read.")
    return parser.parse_args()

def main():
    args = arg_parser()
    txt_file_path = args.txt_file_path
    extracted_text = read_txt_files(txt_file_path)

    # Initialize OpenAI client
    openai_client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=api_key
)

    # Create a simple RAG pipeline
    response = openai_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Here is some text: {extracted_text}. What can you tell me about it?"}
        ]
    )

    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
