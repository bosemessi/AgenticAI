import os
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('NEBIUS_API_KEY')

from src.datareader_utils.datareader import read_txt_files
from src.preprocessor_utils.simple_chunking import simple_overlapping_chunking
from src.helper_funcs.similarity import semantic_search

class simpleRAG:
    def __init__(self, txt_file_path, user_query, chunk_size=1000, overlap=200, embdedding_model="BAAI/bge-en-icl", response_model="meta-llama/Meta-Llama-3.1-70B-Instruct", client=None):
        self.txt_file_path = txt_file_path
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
        extracted_text = read_txt_files(self.txt_file_path)
        chunks = simple_overlapping_chunking(extracted_text, chunk_size=self.chunk_size, overlap=self.overlap)
        logger.info(f"Number of chunks: {len(chunks)}")

        ## Take user query and search for relevant chunks
        top_k_indices = semantic_search(self.user_query, chunks, k=5, model=self.embedding_model, client=self.client)
        logger.info(f"Top K indices: {top_k_indices}")
        top_chunks = [chunks[idx] for idx in top_k_indices]

        ## Create the user prompt based on the top chunks
        user_prompt = "\n".join([f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
        user_prompt = f"{user_prompt}\nQuestion: {self.user_query}"
        logger.info(f"User Prompt: {user_prompt}")

        ## Call OpenAI model with the user prompt and context
        response = self.client.chat.completions.create(
            model=self.response_model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers strictly based on the given context. If the context does not contain the answer, say 'I don't know'."},
                {"role": "user", "content": user_prompt}
            ]    
        )

        ## Extract the AI answer from the response
        ai_answer = response.choices[0].message.content.strip()
        return ai_answer
