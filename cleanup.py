# Contains all functions that deal with stop word removal.
import json
import os

from document import Document
import re
from collections import Counter
DATA_PATH = 'data'
STOPWORD_FILE_PATH = os.path.join(DATA_PATH, 'stopwords.json')


STOP_WORD_LIST = []
def remove_symbols(text_string: str) -> str:
    """
    Removes all punctuation marks and similar symbols from a given string.
    Occurrences of "'s" are removed as well.
    :param text:
    :return:
    """

    # TODO: Implement this function. (PR02)
    # Remove all punctuation marks except apostrophes within words and occurrences of "'s"
    text_string = re.sub(r'(?<!\w)\'|\'(?!\w)|\'s\b|[^\w\s\']', '', text_string)

    # Replace line breaks and multiple spaces with single space
    text_string = re.sub(r'[\n\s]+', ' ', text_string)

    return text_string.strip()


def is_stop_word(term: str, stop_word_list: list[str]) -> bool:
    """
    Checks if a given term is a stop word.
    :param stop_word_list: List of all considered stop words.
    :param term: The term to be checked.
    :return: True if the term is a stop word.
    """
    # TODO: Implement this function  (PR02)
    return term.lower() in stop_word_list


def remove_stop_words_from_term_list(term_list: list[str]) -> list[str]:
    """
    Takes a list of terms and removes all terms that are stop words.
    :param term_list: List that contains the terms
    :return: List of terms without stop words
    """
    # Hint:  Implement the functions remove_symbols() and is_stop_word() first and use them here.
    # Open the JSON file containing stop words and load them into a list
    with open(STOPWORD_FILE_PATH, "r") as json_file:
        stop_words = json.load(json_file)
    return [term for term in term_list if not is_stop_word(term,stop_words)]


def filter_collection(collection: list[Document]):
    """
    For each document in the given collection, this method takes the term list and filters out the stop words.
    Warning: The result is NOT saved in the documents term list, but in an extra field called filtered_terms.
    :param collection: Document collection to process
    """
    # Hint:  Implement remove_stop_words_from_term_list first and use it here.
    # TODO: Implement this function. (PR02)
    for document in collection:
        # Clean and lowercase the raw text of the document
        cleaned_text = remove_symbols(document.raw_text).lower()
        # Split the cleaned text into individual terms
        terms = cleaned_text.split()
        # Filter out stop words
        document.filtered_terms = remove_stop_words_from_term_list(terms)


def load_stop_word_list(raw_file_path: str) -> list[str]:
    """
    Loads a text file that contains stop words and saves it as a list. The text file is expected to be formatted so that
    each stop word is in a new line, e. g. like englishST.txt
    :param raw_file_path: Path to the text file that contains the stop words
    :return: List of stop words
    """
    # TODO: Implement this function. (PR02)
    global STOP_WORD_LIST
    # Open the text file containing stop words
    with open(raw_file_path, 'r') as file:
        STOP_WORD_LIST = [line.strip().lower() for line in file]
    return STOP_WORD_LIST


def create_stop_word_list_by_frequency(collection: list[Document]) -> list[str]:
    """
    Uses the method of J. C. Crouch (1990) to generate a stop word list by finding high and low frequency terms in the
    provided collection.
    :param collection: Collection to process
    :return: List of stop words
    """
    # TODO: Implement this function. (PR02)
    global STOP_WORD_LIST

    LOW_FREQ_THRESHOLD = 1
    HIGH_FREQ_RATIO = 0.5  # Ratio of documents in which a term appears

    term_freq = Counter() # Counter to store term frequencies
    doc_freq = Counter()  # Document frequency

    # Iterate over each document in the collection
    for doc in collection:
        cleaned_text = remove_symbols(doc.raw_text).lower()
        terms = cleaned_text.split() # Split the text into individual terms
        unique_terms = set(terms)  # To count each term only once per document
        term_freq.update(terms)
        doc_freq.update(unique_terms)

    num_documents = len(collection)
    high_freq_threshold = num_documents * HIGH_FREQ_RATIO
    # Generate stop words based on term frequencies and document frequencies
    stop_words = [
        term for term, freq in term_freq.items()
        if freq <= LOW_FREQ_THRESHOLD or doc_freq[term] >= high_freq_threshold
    ]
    STOP_WORD_LIST = stop_words
    return stop_words