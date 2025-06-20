from typing import List
import numpy as np
from openai import OpenAI
from src.preprocessor_utils.embeddings import create_embeddings

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        vec_a (np.ndarray): First vector.
        vec_b (np.ndarray): Second vector.
        
    Returns:
        float: Cosine similarity between vec_a and vec_b.
    """
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def semantic_search(query: str, chunks: List[str], k: int, model: str, client: OpenAI) -> List[int]:
    """
    Perform semantic search to find the top k chunks most similar to the query.
    
    Args:
        query (str): User Query string.
        chunks (list of str): List of chunk strings from the knowledge text.
        k (int): Number of top results to return.
        model (str): Model to use for creating embeddings.
        client (OpenAI): OpenAI client to use for API requests.
        
    Returns:
        list: Indices of the top k chunks most similar to the query.
    """
    query_embedding_response = create_embeddings([query], model, client)
    query_embedding = query_embedding_response.data[0].embedding
    chunk_embedding_response = create_embeddings(chunks, model, client)
    chunk_embeddings = [chunk.embedding for chunk in chunk_embedding_response.data]
    similarities = []
    for chunk_embedding in chunk_embeddings:
        similarity = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding))
        similarities.append(similarity)

    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    return top_k_indices

def context_enriched_search(query: str, chunks: List[str], k: int, model: str, client: OpenAI, n_context: int) -> List[str]:
    """
    Perform context-enriched search to find the top k chunks most similar to the query,
    enriched with additional context from the n_context most similar chunks.
    
    Args:
        query (str): User Query string.
        chunks (list of str): List of chunk strings from the knowledge text.
        k (int): Number of top results to return.
        model (str): Model to use for creating embeddings.
        client (OpenAI): OpenAI client to use for API requests.
        n_context (int): Number of context chunks to include in the enriched search.
        
    Returns:
        list: List of enriched context strings for the query.
    """
    top_index = semantic_search(query, chunks, 1, model, client)
    enriched_contexts = []
    
    start = max(0, top_index[0] - n_context)
    end = min(len(chunks), top_index[0] + n_context + 1)
    for idx in range(start, end):
        enriched_contexts.append(chunks[idx])
    
    return enriched_contexts
