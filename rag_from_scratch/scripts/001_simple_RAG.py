import pymupdf
import os
import numpy as np
import json
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('NEBIUS_API_KEY')

import sys

from src.datareader_utils.datareader import read_txt_files
from src.preprocessor_utils.simple_chunking import simple_overlapping_chunking
from src.helper_funcs.similarity import semantic_search
from argparse import ArgumentParser

def arg_parser():
    parser = ArgumentParser(description="Run a simple RAG pipeline with OpenAI.")
    parser.add_argument("--txt_file_path", type=str, required=True, help="Path to the text file to read.")
    parser.add_argument("--user_query", type=str, required=True, help="User query to answer.")
    return parser.parse_args()




def main():
    args = arg_parser()

    # Initialize OpenAI client
    openai_client = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=api_key
    )

    ## Read text file and chunk it
    txt_file_path = args.txt_file_path
    extracted_text = read_txt_files(txt_file_path)
    chunks = simple_overlapping_chunking(extracted_text, chunk_size=1000, overlap=200)
    logger.info(f"Numbe of chunks: {len(chunks)}")
    # logger.info(f"First chunk: {chunks[0]}")

    ## Take user query and search for relevant chunks
    query = args.user_query
    top_k_indices = semantic_search(query, chunks, k=5, model="BAAI/bge-en-icl", client=openai_client)
    logger.info(f"Top K indices: {top_k_indices}")
    # for i, idx in enumerate(top_k_indices):
    #     logger.info(f"Top {i+1} chunk: {chunks[idx]}")
    
    top_chunks = [chunks[idx] for idx in top_k_indices]
    # Create the user prompt based on the top chunks
    user_prompt = "\n".join([f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
    user_prompt = f"{user_prompt}\nQuestion: {query}"
    logger.info(f"User Prompt: {user_prompt}")

    # Create a simple RAG pipeline
    response = openai_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers strictly based on the given context. If the context does not contain the answer, say 'I don't know'."},
            {"role": "user", "content": user_prompt}
        ]
    )
    ai_answer = response.choices[0].message.content.strip()
    logger.info(f"AI Answer: {ai_answer}")
    
if __name__ == "__main__":
    main()
