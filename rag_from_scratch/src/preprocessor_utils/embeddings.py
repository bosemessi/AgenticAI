def create_embeddings(text, model, client):
    """
    Creates embeddings for the given text using the specified model and client.

    Args:
        text (str): The text to be embedded.
        model (str): The model to use for embedding.
        client: The client to use for making API requests.

    Returns:
        list: A list of embeddings for the text.
    """
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response