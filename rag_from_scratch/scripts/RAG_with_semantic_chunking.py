import os
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('NEBIUS_API_KEY')
import numpy as np
from src.datareader_utils.datareader import read_txt_files
from src.preprocessor_utils.embeddings import create_embeddings
from src.helper_funcs.similarity import cosine_similarity, semantic_search
from src.preprocessor_utils.chunking_with_breakpoints import compute_breakpoints

class ragWithSemanticChunking:
    def __init__(self, txt_file_path, user_query, embdedding_model="BAAI/bge-en-icl", response_model="meta-llama/Meta-Llama-3.1-70B-Instruct", client=None):
        self.txt_file_path = txt_file_path
        self.user_query = user_query
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
        ## Read text file and break it into sentences
        extracted_text = read_txt_files(self.txt_file_path)
        sentences = extracted_text.split('. ')
        logger.info(f"Number of sentences: {len(sentences)}")
        
        ## Create sentence embeddings
        sentence_embeddings = create_embeddings(text=sentences,model=self.embedding_model,client=self.client)
        sentence_embeddings = [np.array(embedding.embedding) for embedding in sentence_embeddings.data]

        ## Take consecutive sentences and perform similarity calculation
        sentence_similarities = [cosine_similarity(sentence_embeddings[i], sentence_embeddings[i + 1]) for i in range(len(sentence_embeddings) - 1)]
        logger.info(f"Sentence similarities calculated: {sentence_similarities}")

        ## Compute breakpoints based on the similarity scores
        breakpoints = compute_breakpoints(similarities=sentence_similarities, method="IQR", threshold=70)
        logger.info(f"Breakpoints computed: {breakpoints}")
        
        ## Create chunks based on the breakpoints
        chunks = []
        start_idx = 0
        for bp in breakpoints:
            chunks.append(' '.join(sentences[start_idx:bp + 1]))
            start_idx = bp + 1
        chunks.append('. '.join(sentences[start_idx:]))
        logger.info(f"Number of chunks created: {len(chunks)}")
        
        ## Perform semantic search to find the top k chunks most similar to the user query
        k = 5
        top_k_indices = semantic_search(query=self.user_query, chunks=chunks, k=k, model=self.embedding_model, client=self.client)
        logger.info(f"Top {k} indices found: {top_k_indices}")
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
