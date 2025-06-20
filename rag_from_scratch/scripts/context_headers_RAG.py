import os
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('NEBIUS_API_KEY')

from src.datareader_utils.datareader import read_pdf_files
from src.preprocessor_utils.simple_chunking import simple_overlapping_chunking
from src.preprocessor_utils.generate_chunk_headers import contextual_chunk_headers
from src.helper_funcs.similarity import semantic_search

class contextHeadersRAG:
    def __init__(self, pdf_file_path, user_query, chunk_size=1000, overlap=200, embdedding_model="BAAI/bge-en-icl", response_model="meta-llama/Meta-Llama-3.1-70B-Instruct", client=None):
        self.pdf_file_path = pdf_file_path
        self.user_query = user_query
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedding_model = embdedding_model
        self.response_model = response_model
        if client is None:
            self.client = OpenAI(
                base_url="https://api.studio.nebius.com/v1/",
                api_key=api_key
            )
        else:
            self.client = client

    def pipeline(self):
        ## Read text file and chunk it
        extracted_text = read_pdf_files(self.pdf_file_path)
        chunks = simple_overlapping_chunking(extracted_text, chunk_size=self.chunk_size, overlap=self.overlap)
        logger.info(f"Number of chunks: {len(chunks)}")
        
        ## Generate contextual headers for the chunks
        dict_of_chunks = contextual_chunk_headers(chunks, model=self.response_model, client=self.client)
        
