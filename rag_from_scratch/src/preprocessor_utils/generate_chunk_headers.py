def contextual_chunk_headers(chunks: list[str], model, client) -> dict[str, str]:
    """
    Generate a header for a given chunk of text using a language model.
    
    Args:
        chunks (list[str]): The list of chunks to generate headers for.
        model (str): The model to use for generating the header.
        client: The OpenAI client instance.
    
    Returns:
        dict_of_chunks (dict[str, str]): dict of the generated header for the chunk.
    """
    dict_of_chunks = {}
   
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates concise headers for text chunks."},
            {
                "role": "user", 
                "content": f"There are {len(chunks)} chunks given to you. Generate a list of {len(chunks)} concise and informative headers for each chunk in the following list of text chunks:\n\n{chunks}"}
        ]
    )
    
    headers = response.choices[0].message.content.strip().split(". ")[1:]
    headers = [header.split("**")[1] for header in headers]
    print(f"Number of headers generated: {len(headers)}")
    print(f"Generated headers: {headers}")
    for i, chunk in enumerate(chunks):
        dict_of_chunks[chunk] = headers[i] if i < len(headers) else "No header generated"
    return dict_of_chunks