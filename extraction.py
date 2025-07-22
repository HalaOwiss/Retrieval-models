# Contains functions that deal with the extraction of documents from a text file (see PR01)

import json
import os
import re

from document import Document

def extract_terms(text):
    """
    Extract terms (words) from the given text.
    :param text: Input text
    :return: List of terms
    """
    # Replace contractions like "it's" to "it is"
    text = re.sub(r"\b([a-zA-Z]+)'s\b", r"\1 is", text)
    # Remove punctuation except apostrophes
    text = re.sub(r"[^\w\s']", "", text)
    # Convert text to lowercase and split into terms
    terms = text.lower().split()
    return terms

def extract_collection(source_file_path: str) -> list[Document]:
    """
    Loads a text file (aesopa10.txt) and extracts each of the listed fables/stories from the file.
    :param source_file_name: File name of the file that contains the fables
    :return: List of Document objects
    """

    # TODO: Implement this function. (PR02)
    catalog = []  # This list will store the Document objects.

    splitter = "\n\n\n\n"
    with open(source_file_path, "r") as data_file:
        data = data_file.read().split(splitter)
        # Index for assigning document IDs
        ind = 0
        for index, fable in enumerate(data):
            # Exclude the unwanted sections
            if "This is the SECOND Project Gutenberg Etext of Aesop's Fables" in fable:
                continue
            if "***This edition is being officially released on March 8, 1992***" in fable:
                continue
            if "AESOP'S FABLES (82 Fables)" in fable:
                continue
            raw_text = fable.replace('\n', ' ')  # Full text without line breaks
            terms = extract_terms(raw_text) # Extract terms from the fable text
            parts = fable.split('\n\n\n', 1)
            title = parts[0].strip()
            text_content = parts[1] if len(parts) > 1 else ""
            # Create Document objects for each fable and add them to catalog
            document = Document()
            document.document_id = ind
            document.title = title.strip()
            document.raw_text = text_content.replace('\n', ' ')
            document.terms = terms

            catalog.append(document)
            ind = ind + 1


    return catalog


def save_collection_as_json(collection: list[Document], file_path: str) -> None:
    """
    Saves the collection to a JSON file.
    :param collection: The collection to store (= a list of Document objects)
    :param file_path: Path of the JSON file
    """

    serializable_collection = []
    for document in collection:
        serializable_collection += [{
            'document_id': document.document_id,
            'title': document.title,
            'raw_text': document.raw_text,
            'terms': document.terms,
            'filtered_terms': document.filtered_terms,
            'stemmed_terms': document.stemmed_terms,
            'stemmed_filtered_terms': document.stemmed_filtered_terms
        }]

    with open(file_path, "w") as json_file:
        json.dump(serializable_collection, json_file)


def load_collection_from_json(file_path: str) -> list[Document]:
    """
    Loads the collection from a JSON file.
    :param file_path: Path of the JSON file
    :return: list of Document objects
    """
    try:
        with open(file_path, "r") as json_file:
            json_collection = json.load(json_file)

        collection = []
        for doc_dict in json_collection:
            document = Document()
            document.document_id = doc_dict.get('document_id')
            document.title = doc_dict.get('title')
            document.raw_text = doc_dict.get('raw_text')
            document.terms = doc_dict.get('terms')
            document.filtered_terms = doc_dict.get('filtered_terms')
            document.stemmed_terms = doc_dict.get('stemmed_terms')
            document.stemmed_filtered_terms = doc_dict.get('stemmed_filtered_terms')
            collection += [document]

        return collection
    except FileNotFoundError:
        print('No collection was found. Creating empty one.')
        return []
