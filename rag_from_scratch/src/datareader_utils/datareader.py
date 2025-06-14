import pymupdf

def read_txt_files(txt_file_path:str) -> str:
    """
    Extracts text from a .txt file.
    Args:
        txt_file_path (str): Path to the .txt file.
    Returns:
        str: The content of the .txt file.
    """
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def read_pdf_files(pdf_file_path:str) -> str:
    """
    Extracts text from a .pdf file.
    Args:
        pdf_file_path (str): Path to the .pdf file.
    Returns:
        str: The content of the .pdf file.
    """
    doc = pymupdf.open(pdf_file_path)
    content = ""
    for page in doc:
        content += page.get_text()
    doc.close()
    return content
