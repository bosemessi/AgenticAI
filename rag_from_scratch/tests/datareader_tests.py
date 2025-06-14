def read_txt_files_test(txt_file_path: str):
    """
    Tests whether the txt_file_path is a proper str ending with '.txt'.
    Also tests whether the file exists and is readable.
    Also tests whether the function returns a string.
    This is a test function for read_txt_files.
    Args:
        txt_file_path (str): Path to the .txt file.
    """
    assert isinstance(txt_file_path, str), "txt_file_path must be a string"
    assert txt_file_path.endswith('.txt'), "txt_file_path must end with '.txt'"
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        assert isinstance(content, str), "Content must be a string"
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {txt_file_path} does not exist or is not readable.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {e}")
    
