def simple_overlapping_chunking(text, chunk_size=1000, overlap=200):
    """
    Splits the text into overlapping chunks.

    Args:
        text (str): The text to be chunked.
        chunk_size (int): The size of each chunk.
        overlap (int): The number of characters to overlap between chunks.

    Returns:
        list: A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks