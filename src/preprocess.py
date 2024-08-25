import pandas as pd
import re
from contractions import contractions_dict
import string
import unicodedata
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Load the dataset
# df = pd.read_csv('/Users/rohitrawat/job-prep/Assignments/accrete-ai/text-summarization/data/processed/news_summary.csv')

# Function to expand contractions
def expand_contractions(text, contraction_map=contractions_dict):
    """
    Expands contractions in the given text using a contraction map.

    Args:
        text (str): The input text containing contractions.
        contraction_map (dict): A dictionary mapping contractions to their expanded forms. Defaults to contractions_dict.

    Returns:
        str: The text with expanded contractions.

    """
    # Rest of the code...
    # Using regex for getting all contracted words
    contractions_keys = '|'.join(contraction_map.keys())
    contractions_pattern = re.compile(f'({contractions_keys})', flags=re.DOTALL)

    def expand_match(contraction):
        # Getting entire matched sub-string
        match = contraction.group(0)
        expanded_contraction = contraction_map.get(match)
        if not expanded_contraction:
            return match
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

# Remove puncuation from word
def rm_punc_from_word(word):
    """
    Removes punctuation from a given word.

    Parameters:
    word (str): The word from which punctuation needs to be removed.

    Returns:
    str: The word without any punctuation.
    """
    clean_alphabet_list = [
        alphabet for alphabet in word if alphabet not in string.punctuation
    ]
    return ''.join(clean_alphabet_list)

# Remove puncuation from text
def rm_punc_from_text(text):
    """
    Removes punctuation from the given text.

    Parameters:
    text (str): The input text.

    Returns:
    str: The text with punctuation removed.
    """
    clean_word_list = [rm_punc_from_word(word) for word in text]
    return ''.join(clean_word_list)

# Remove numbers from text
def rm_number_from_text(text):
    """
    Remove numbers from the given text.

    Parameters:
    text (str): The input text.

    Returns:
    str: The text with numbers removed.
    """
    text = re.sub('[0-9]+', '', text)
    return ' '.join(text.split())  # to rm `extra` white space

# Remove stopwords from text
def rm_stopwords_from_text(text):
    """
    Removes stopwords from the given text.

    Parameters:
    text (str): The input text to remove stopwords from.

    Returns:
    str: The text with stopwords removed.
    """
    _stopwords = stopwords.words('english')
    text = text.split()
    word_list = [word for word in text if word not in _stopwords]
    return ' '.join(word_list)

# Function to remove accented characters
def remove_accented_chars(text):
    """
    Removes accented characters from the given text.

    Parameters:
    text (str): The input text.

    Returns:
    str: The text with accented characters removed.
    """
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

# Function to remove special characters
def remove_special_characters(text, remove_digits=False):
    """
    Remove special characters from the given text.

    Parameters:
    - text (str): The input text to remove special characters from.
    - remove_digits (bool): Whether to remove digits as well. Default is False.

    Returns:
    - str: The text with special characters removed.
    """
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text

# Function to remove stopwords
def remove_stopwords(text):
    """
    Remove stopwords from the given text.

    Parameters:
    text (str): The input text to remove stopwords from.

    Returns:
    str: The filtered text with stopwords removed.
    """
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def more_cleaning(text):
    """
    Cleans the given text by removing unnecessary characters, extra white spaces, accented characters,
    and replacing specific patterns with placeholders.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    # code implementation goes here
    # there are hyphen(–) in many titles, so replacing it with empty str
    # this hyphen(–) is different from normal hyphen(-)
    text = re.sub('–', '', text)
    text = ' '.join(text.split())  # removing `extra` white spaces

    # Removing unnecessary characters from text
    text = re.sub("(\\t)", ' ', str(text)).lower()
    text = re.sub("(\\r)", ' ', str(text)).lower()
    text = re.sub("(\\n)", ' ', str(text)).lower()

    # remove accented chars ('Sómě Áccěntěd těxt' => 'Some Accented text')
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode(
        'utf-8', 'ignore'
    )

    text = re.sub("(__+)", ' ', str(text)).lower()
    text = re.sub("(--+)", ' ', str(text)).lower()
    text = re.sub("(~~+)", ' ', str(text)).lower()
    text = re.sub("(\+\++)", ' ', str(text)).lower()
    text = re.sub("(\.\.+)", ' ', str(text)).lower()

    text = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(text)).lower()

    text = re.sub("(mailto:)", ' ', str(text)).lower()
    text = re.sub(r"(\\x9\d)", ' ', str(text)).lower()
    text = re.sub("([iI][nN][cC]\d+)", 'INC_NUM', str(text)).lower()
    text = re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM',
                  str(text)).lower()

    text = re.sub("(\.\s+)", ' ', str(text)).lower()
    text = re.sub("(\-\s+)", ' ', str(text)).lower()
    text = re.sub("(\:\s+)", ' ', str(text)).lower()
    text = re.sub("(\s+.\s+)", ' ', str(text)).lower()

    try:
        url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', str(text))
        repl_url = url.group(3)
        text = re.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', repl_url, str(text))
    except Exception as e:
        pass

    text = re.sub("(\s+)", ' ', str(text)).lower()
    text = re.sub("(\s+.\s+)", ' ', str(text)).lower()

    return text

# Example usage of the functions

# Example text
text = "I can't wait to see the new movie. It's gonna be awesome!"

# Example usage of expand_contractions()
expanded_text = expand_contractions(text)
print(expanded_text)
# Output: "I cannot wait to see the new movie. It is gonna be awesome!"

# Example usage of rm_punc_from_word()
word = "hello!"
clean_word = rm_punc_from_word(word)
print(clean_word)
# Output: "hello"

# Example usage of rm_punc_from_text()
clean_text = rm_punc_from_text(text)
print(clean_text)
# Output: "I cant wait to see the new movie Its gonna be awesome"

# Example usage of rm_number_from_text()
text_with_numbers = "I have 5 apples and 3 oranges"
clean_text = rm_number_from_text(text_with_numbers)
print(clean_text)
# Output: "I have apples and oranges"

# Example usage of rm_stopwords_from_text()
text_with_stopwords = "I have a cat and a dog"
clean_text = rm_stopwords_from_text(text_with_stopwords)
print(clean_text)
# Output: "I cat dog"

# Example usage of remove_accented_chars()
text_with_accents = "Sómě Áccěntěd těxt"
clean_text = remove_accented_chars(text_with_accents)
print(clean_text)
# Output: "Some Accented text"

# Example usage of remove_special_characters()
text_with_special_chars = "Hello! How are you?"
clean_text = remove_special_characters(text_with_special_chars)
print(clean_text)
# Output: "Hello How are you"

# Example usage of remove_stopwords()
text_with_stopwords = "I have a cat and a dog"
clean_text = remove_stopwords(text_with_stopwords)
print(clean_text)
# Output: "cat dog"

# Example usage of more_cleaning()
dirty_text = "This is a dirty text with unnecessary characters!!!"
clean_text = more_cleaning(dirty_text)
print(clean_text)
# Output: "this is a dirty text with unnecessary characters"