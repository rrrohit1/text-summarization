import pandas as pd
from transformers import pipeline

# Initialize the summarizer
summarizer = pipeline("summarization")

# Function to split text into chunks
def split_text_into_chunks(text, max_chunk_size=512):
    """
    Splits the given text into chunks of maximum size 'max_chunk_size'.
    Parameters:
    - text (str): The text to be split into chunks.
    - max_chunk_size (int): The maximum size of each chunk. Default is 512.
    Returns:
    - chunks (list): A list of chunks, where each chunk is a string.
    Example:
    >>> text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    >>> split_text_into_chunks(text, max_chunk_size=10)
    ['Lorem ipsum', 'dolor sit', 'amet,', 'consectetur', 'adipiscing', 'elit.']
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Function to summarize long text
def summarize_long_text(text, max_chunk_size=512, max_length=150, min_length=30):
    """
    Summarizes a long text by splitting it into chunks and generating summaries for each chunk.
    Args:
        text (str): The long text to be summarized.
        max_chunk_size (int, optional): The maximum size of each chunk in characters. Defaults to 512.
        max_length (int, optional): The maximum length of each summary in tokens. Defaults to 150.
        min_length (int, optional): The minimum length of each summary in tokens. Defaults to 30.
    Returns:
        str: The summarized text.
    """
    chunks = split_text_into_chunks(text, max_chunk_size)
    summaries = []
    
    for chunk in chunks:
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    
    return " ".join(summaries)

# Apply the summarization to the DataFrame
df['generated_summary'] = df['text'].apply(summarize_long_text)

# Example usage
output_2 = summarize_long_text(df.text.values[2])
print(output_2)