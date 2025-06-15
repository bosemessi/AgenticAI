import numpy as np

def compute_breakpoints(similarities: list, method="percentile", threshold=70)-> list:
    """
    Compute breakpoints based on the similarity scores between consecutive sentences.
    
    Args:
        similarities (list): List of similarity scores between consecutive sentences.
        method (str): Method to compute breakpoints -> "percentile", "stddev", "IQR
        threshold (int): Threshold value for the percentile method.
        
    Returns:
        list: List of indices where breakpoints occur.
    """
    if method == "percentile":
        threshold_value = np.percentile(similarities, threshold)
        breakpoints = [i for i, sim in enumerate(similarities) if sim < threshold_value]
    elif method == "stddev":
        mean = np.mean(similarities)
        std_dev = np.std(similarities)
        threshold_value = mean - threshold*std_dev
        breakpoints = [i for i, sim in enumerate(similarities) if sim < threshold]
    elif method == "IQR":
        q1 = np.percentile(similarities, 25)
        q3 = np.percentile(similarities, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        breakpoints = [i for i, sim in enumerate(similarities) if sim < lower_bound]
    else:
        raise ValueError("Method must be either 'percentile' or 'stddev' or 'IQR'.")
    
    return breakpoints