# --------------------------------------------------------------------------------
# Information Retrieval SS2024 - Practical Assignment Template
# --------------------------------------------------------------------------------
# This Python template is provided as a starting point for your assignments PR02-04.
# It serves as a base for a very rudimentary text-based information retrieval system.
#
# Please keep all instructions from the task description in mind.
# Especially, avoid changing the base structure, function or class names or the
# underlying program logic. This is necessary to run automated tests on your code.
#
# Instructions:
# 1. Read through the whole template to understand the expected workflow and outputs.
# 2. Implement the required functions and classes, filling in your code where indicated.
# 3. Test your code to ensure functionality and correct handling of edge cases.
#
# Good luck!


import json
import os
from collections import deque
import cleanup
import extraction
import models
import porter
import time
from typing import List, Dict, Set
import re
import math
from collections import defaultdict, Counter
from document import Document

# Important paths:
RAW_DATA_PATH = 'raw_data'
DATA_PATH = 'data'
COLLECTION_PATH = os.path.join(DATA_PATH, 'my_collection.json')
STOPWORD_FILE_PATH = os.path.join(DATA_PATH, 'stopwords.json')

# Menu choices:
(CHOICE_LIST, CHOICE_SEARCH, CHOICE_EXTRACT, CHOICE_UPDATE_STOP_WORDS, CHOICE_SET_MODEL, CHOICE_SHOW_DOCUMENT,
 CHOICE_EXIT) = 1, 2, 3, 4, 5, 6, 9
MODEL_BOOL_LIN, MODEL_BOOL_INV, MODEL_BOOL_SIG, MODEL_FUZZY, MODEL_VECTOR = 1, 2, 3, 4, 5
SW_METHOD_LIST, SW_METHOD_CROUCH = 1, 2


class InformationRetrievalSystem(object):
    def __init__(self):
        if not os.path.isdir(DATA_PATH):
            os.makedirs(DATA_PATH)

        # Collection of documents, initially empty.
        try:
            self.collection = extraction.load_collection_from_json(COLLECTION_PATH)
        except FileNotFoundError:
            print('No previous collection was found. Creating empty one.')
            self.collection = []

        # Stopword list, initially empty.
        try:
            with open(STOPWORD_FILE_PATH, 'r') as f:
                self.stop_word_list = json.load(f)
        except FileNotFoundError:
            print('No stopword list was found.')
            self.stop_word_list = []

        self.model = None  # Saves the current IR model in use.
        self.output_k = 10  # Controls how many results should be shown for a query.


    def main_menu(self):
        """
        Provides the main loop of the CLI menu that the user interacts with.
        """
        while True:
            print(f'Current retrieval model: {self.model}')
            print(f'Current collection: {len(self.collection)} documents')
            print()
            print('Please choose an option:')
            print(f'{CHOICE_LIST} - List documents')
            print(f'{CHOICE_SEARCH} - Search for term')
            print(f'{CHOICE_EXTRACT} - Build collection')
            print(f'{CHOICE_UPDATE_STOP_WORDS} - Rebuild stopword list')
            print(f'{CHOICE_SET_MODEL} - Set model')
            print(f'{CHOICE_SHOW_DOCUMENT} - Show a specific document')
            print(f'{CHOICE_EXIT} - Exit')
            try:
                action_choice = int(input('Enter choice: '))
            except ValueError:
                print("Invalid input. Please enter a valid integer corresponding to the action choice.")
                continue
            # action_choice = int(input('Enter choice: '))

            if action_choice == CHOICE_LIST:
                # List documents in CLI.
                if self.collection:
                    for document in self.collection:
                        print(document)
                else:
                    print('No documents.')
                print()

            elif action_choice == CHOICE_SEARCH:
                if self.model is None:
                    print("No retrieval model selected. Please choose a model first.")
                    continue
                # Read a query string from the CLI and search for it.

                # Determine desired search parameters:
                SEARCH_NORMAL, SEARCH_SW, SEARCH_STEM, SEARCH_SW_STEM = 1, 2, 3, 4
                print('Search options:')
                print(f'{SEARCH_NORMAL} - Standard search (default)')
                print(f'{SEARCH_SW} - Search documents with removed stopwords')
                print(f'{SEARCH_STEM} - Search documents with stemmed terms')
                print(f'{SEARCH_SW_STEM} - Search documents with removed stopwords AND stemmed terms')
                try:
                    search_mode = int(input('Enter choice: '))
                except ValueError:
                    print("Invalid input. Please enter a valid integer corresponding to the action choice.")
                    continue
                stop_word_filtering = (search_mode == SEARCH_SW) or (search_mode == SEARCH_SW_STEM)
                stemming = (search_mode == SEARCH_STEM) or (search_mode == SEARCH_SW_STEM)

                # Actual query processing begins here:
                query = input('Query: ')
                userQuery = query
                start_time = time.time()
                if stemming:
                    query = porter.stem_query_terms(query)
                    print('stem',query)

                if isinstance(self.model, models.InvertedListBooleanModel):
                    results = self.inverted_list_search(query, stemming, stop_word_filtering)
                elif isinstance(self.model, models.VectorSpaceModel):
                    results = self.buckley_lewit_search(query, stemming, stop_word_filtering,gamma=10)
                elif isinstance(self.model, models.SignatureBasedBooleanModel):
                    results = self.signature_search(query, stemming, stop_word_filtering)
                else:
                    results = self.basic_query_search(query, stemming, stop_word_filtering)
                end_time = time.time()
                elapsed_time_ms = (end_time - start_time) * 1000
                # Output of results:
                for (score, document) in results:
                    print(f'{score}: {document}')

                # Output of quality metrics:
                print()
                print(f'precision: {self.calculate_precision(userQuery,results)}')
                print(f'recall: {self.calculate_recall(userQuery,results)}')
                print(f"Time taken for query processing: {elapsed_time_ms:.2f} ms")

            elif action_choice == CHOICE_EXTRACT:
                # Extract document collection from text file.

                raw_collection_file = os.path.join(RAW_DATA_PATH, 'aesopa10.txt')
                self.collection = extraction.extract_collection(raw_collection_file)
                assert isinstance(self.collection, list)
                assert all(isinstance(d, Document) for d in self.collection)

                if input('Should stopwords be filtered? [y/N]: ') == 'y':
                    cleanup.filter_collection(self.collection)

                if input('Should stemming be performed? [y/N]: ') == 'y':
                    porter.stem_all_documents(self.collection)

                extraction.save_collection_as_json(self.collection, COLLECTION_PATH)
                print('Done.\n')

            elif action_choice == CHOICE_UPDATE_STOP_WORDS:
                # Rebuild the stop word list, using one out of two methods.

                print('Available options:')
                print(f'{SW_METHOD_LIST} - Load stopword list from file')
                print(f"{SW_METHOD_CROUCH} - Generate stopword list using Crouch's method")
                try:
                    method_choice = int(input('Enter choice: '))
                except ValueError:
                    print("Invalid input. Please enter a valid integer corresponding to the action choice.")
                    continue
                if method_choice in (SW_METHOD_LIST, SW_METHOD_CROUCH):
                    # Load stop words using the desired method:
                    if method_choice == SW_METHOD_LIST:
                        self.stop_word_list = cleanup.load_stop_word_list(os.path.join(RAW_DATA_PATH, 'englishST.txt'))
                        print('Done.\n')
                    elif method_choice == SW_METHOD_CROUCH:
                        self.stop_word_list = cleanup.create_stop_word_list_by_frequency(self.collection)
                        print('Done.\n')

                    # Save new stopword list into file:
                    with open(STOPWORD_FILE_PATH, 'w') as f:
                        json.dump(self.stop_word_list, f)
                else:
                    print('Invalid choice.')

            elif action_choice == CHOICE_SET_MODEL:
                # Choose and set the retrieval model to use for searches.

                print()
                print('Available models:')
                print(f'{MODEL_BOOL_LIN} - Boolean model with linear search')
                print(f'{MODEL_BOOL_INV} - Boolean model with inverted lists')
                print(f'{MODEL_BOOL_SIG} - Boolean model with signature-based search')
                print(f'{MODEL_FUZZY} - Fuzzy set model')
                print(f'{MODEL_VECTOR} - Vector space model')
                try:
                    model_choice = int(input('Enter choice: '))
                except ValueError:
                    print("Invalid input. Please enter a valid integer corresponding to the action choice.")
                    continue
                if model_choice == MODEL_BOOL_LIN:
                    self.model = models.LinearBooleanModel()
                elif model_choice == MODEL_BOOL_INV:
                    self.model = models.InvertedListBooleanModel()
                elif model_choice == MODEL_BOOL_SIG:
                    self.model = models.SignatureBasedBooleanModel()
                elif model_choice == MODEL_FUZZY:
                    self.model = models.FuzzySetModel()
                elif model_choice == MODEL_VECTOR:
                    self.model = models.VectorSpaceModel()
                else:
                    print('Invalid choice.')

            elif action_choice == CHOICE_SHOW_DOCUMENT:
                try:
                    target_id = int(input('ID of the desired document:'))
                except ValueError:
                    print("Invalid input. Please enter a valid integer corresponding to the action choice.")
                    continue
                found = False
                for document in self.collection:
                    if document.document_id == target_id:
                        print(document.title)
                        print('-' * len(document.title))
                        print(document.raw_text)
                        found = True

                if not found:
                    print(f'Document #{target_id} not found!')

            elif action_choice == CHOICE_EXIT:
                break
            else:
                print('Invalid choice.')

            print()
            input('Press ENTER to continue...')
            print()

    def basic_query_search(self, query: str, stemming: bool, stop_word_filtering: bool) -> list:
        """
        Searches the collection for a query string. This method is "basic" in that it does not use any special algorithm
        to accelerate the search. It simply calculates all representations and matches them, returning a sorted list of
        the k most relevant documents and their scores.
        :param query: Query string
        :param stemming: Controls, whether stemming is used
        :param stop_word_filtering: Controls, whether stop-words are ignored in the search
        :return: List of tuples, where the first element is the relevance score and the second the corresponding
        document
        """
        query_representation = self.model.query_to_representation(query)
        document_representations = [self.model.document_to_representation(d, stop_word_filtering, stemming)
                                    for d in self.collection]
        scores = [self.model.match(dr, query_representation) for dr in document_representations]
        ranked_collection = sorted(zip(scores, self.collection), key=lambda x: x[0], reverse=True)
        filtered_ranked_collection = [(score, doc) for score, doc in ranked_collection if score > 0]
        results = ranked_collection[:self.output_k]
        # scores = [(1.0, doc) for doc in self.collection if doc.document_id in result_set]
        # return scores
        return filtered_ranked_collection

    def read_ground_truth(self,file_path: str) -> Dict[str, Set[int]]:
        ground_truth = {}
        with open(file_path, 'r') as file:
            for line in file:
                # Skip lines starting with '#' or empty lines
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split(' - ')
                    if len(parts) == 2:
                        term, doc_ids = parts
                        # Check if doc_ids is not empty
                        if doc_ids.strip():
                            # Adjust IDs to be zero-based
                            ground_truth[term] = set(map(lambda x: int(x) - 1, doc_ids.split(',')))
                    else:
                        print(f"Ignoring invalid line in ground_truth.txt: {line.strip()}")

        return ground_truth

    def inverted_list_search(self, query: str, stemming: bool, stop_word_filtering: bool) -> list:
        """
        Fast Boolean query search for inverted lists.
        :param query: Query string
        :param stemming: Controls, whether stemming is used
        :param stop_word_filtering: Controls, whether stop-words are ignored in the search
        :return: List of tuples, where the first element is the relevance score and the second the corresponding
        document
        """
        result_set=set()
        if stemming == False and stop_word_filtering == False:
            if self.model.inverted_list is None:
                self.model.build_inverted_list(self.collection, stop_word_filtering, stemming)
            query_terms = self.model.query_to_representation(query)
            # print(query_terms)
            result_set = self.model.evaluate_query(query_terms, self.model.inverted_list, self.collection)

        if stemming == False and stop_word_filtering == True:
            if self.model.inverted_list_filtered is None:
                self.model.build_inverted_list(self.collection, stop_word_filtering, stemming)
            query_terms = self.model.query_to_representation(query)
            # print(query_terms)
            result_set = self.model.evaluate_query(query_terms, self.model.inverted_list_filtered,self.collection)

        if stemming == True and stop_word_filtering == False:
            if self.model.inverted_list_stemmed is None:
                self.model.build_inverted_list(self.collection, stop_word_filtering, stemming)
            query_terms = self.model.query_to_representation(query)
            # print(query_terms)
            result_set = self.model.evaluate_query(query_terms, self.model.inverted_list_stemmed,self.collection)

        if stemming == True and stop_word_filtering == True:
            if self.model.inverted_list_stemmed_filtered is None:
                self.model.build_inverted_list(self.collection, stop_word_filtering, stemming)
            query_terms = self.model.query_to_representation(query)
            # print(query_terms)
            result_set = self.model.evaluate_query(query_terms, self.model.inverted_list_stemmed_filtered,self.collection)
            # print(query_terms)

        scores = [(1.0, doc) for doc in self.collection if doc.document_id in result_set]
        return scores

    # def buckley_lewit_search(self, query: str, stemming: bool, stop_word_filtering: bool) -> list:
    #     """
    #     Fast query search for the Vector Space Model using the algorithm by Buckley & Lewit.
    #     :param query: Query string
    #     :param stemming: Controls, whether stemming is used
    #     :param stop_word_filtering: Controls, whether stop-words are ignored in the search
    #     :return: List of tuples, where the first element is the relevance score and the second the corresponding
    #     document
    #     """
    #     # TODO: Implement this function (PR04)
    #     gamma=10
    #     if stop_word_filtering and stemming:
    #         if self.model.inverted_index_filtered_stemmed is None:
    #             self.model.inverted_index_filtered_stemmed = defaultdict(list)
    #             self.model.build_inv_list(self.collection, stop_word_filtering, stemming)
    #             self.model.calculate_idf(stopword_filtering=stop_word_filtering, stemming=stemming)
    #     elif stop_word_filtering:
    #         if self.model.inverted_index_filtered is None:
    #             self.model.inverted_index_filtered = defaultdict(list)
    #             self.model.build_inv_list(self.collection, stop_word_filtering, stemming)
    #             self.model.calculate_idf(stopword_filtering=stop_word_filtering, stemming=stemming)
    #     elif stemming:
    #         if self.model.inverted_index_stemmed is None:
    #             self.model.inverted_index_stemmed = defaultdict(list)
    #             self.model.build_inv_list(self.collection, stop_word_filtering, stemming)
    #             self.model.calculate_idf(stopword_filtering=stop_word_filtering, stemming=stemming)
    #     else:
    #         if self.model.inverted_index is None:
    #             self.model.inverted_index = defaultdict(list)
    #             self.model.build_inv_list(self.collection, stop_word_filtering, stemming)
    #             self.model.calculate_idf(stopword_filtering=stop_word_filtering, stemming=stemming)
    #     # query = stem_query_terms(query)
    #     query_vector = self.model.query_to_representation(query, stopword_filtering=stop_word_filtering, stemming=stemming)
    #     # print('query_vector', query_vector)
    #     scores = defaultdict(float)
    #     top_docs = {}
    #
    #     # Sort query terms by descending weights
    #     sorted_query_terms = sorted(query_vector.items(), key=lambda item: item[1], reverse=True)
    #
    #     # for term, wqk in sorted_query_terms:
    #     #     if wqk <= 0:
    #     #         continue
    #     #     if stop_word_filtering and stemming:
    #     #         for doc_id, wdk in self.model.inverted_index_filtered_stemmed.get(term, []):
    #     #             if doc_id in scores:
    #     #                 scores[doc_id] += wqk * wdk
    #     #             else:
    #     #                 scores[doc_id] = wqk * wdk
    #     #
    #     #             # Insert into top docs if necessary
    #     #             self.model._insert_into_top_docs(top_docs, doc_id, scores[doc_id], gamma)
    #     #     elif stop_word_filtering:
    #     #         for doc_id, wdk in self.model.inverted_index_filtered.get(term, []):
    #     #             if doc_id in scores:
    #     #                 scores[doc_id] += wqk * wdk
    #     #             else:
    #     #                 scores[doc_id] = wqk * wdk
    #     #
    #     #             # Insert into top docs if necessary
    #     #             self.model._insert_into_top_docs(top_docs, doc_id, scores[doc_id], gamma)
    #     #     elif stemming:
    #     #         for doc_id, wdk in self.model.inverted_index_stemmed.get(term, []):
    #     #             if doc_id in scores:
    #     #                 scores[doc_id] += wqk * wdk
    #     #             else:
    #     #                 scores[doc_id] = wqk * wdk
    #     #
    #     #             # Insert into top docs if necessary
    #     #             self.model._insert_into_top_docs(top_docs, doc_id, scores[doc_id], gamma)
    #     #     else:
    #     #         for doc_id, wdk in self.model.inverted_index.get(term, []):
    #     #             if doc_id in scores:
    #     #                 scores[doc_id] += wqk * wdk
    #     #             else:
    #     #                 scores[doc_id] = wqk * wdk
    #     #
    #     #             # Insert into top docs if necessary
    #     #             self.model._insert_into_top_docs(top_docs, doc_id, scores[doc_id], gamma)
    #     #     if self.model._can_terminate_early(top_docs, query_vector, sorted_query_terms, gamma):
    #     #         # print('bye')
    #     #         break
    #     #
    #     # # Return the top gamma results
    #     # # return [(score, doc) for score, doc in sorted(top_docs, key=lambda x: x[0], reverse=True)]
    #     # return [(score, self.model.documents[doc_id]) for doc_id, score in
    #     #         sorted(top_docs.items(), key=lambda x: x[1], reverse=True)]
    #     for term, wqk in sorted_query_terms:
    #         if wqk <= 0:
    #             continue
    #         if stop_word_filtering and stemming:
    #             for doc_id, wdk in self.model.inverted_index_filtered_stemmed.get(term, []):
    #                 scores[doc_id] += wqk * wdk
    #                 self.model._insert_into_top_docs(top_docs, doc_id, scores[doc_id], gamma)
    #         elif stop_word_filtering:
    #             for doc_id, wdk in self.model.inverted_index_filtered.get(term, []):
    #                 scores[doc_id] += wqk * wdk
    #                 self.model._insert_into_top_docs(top_docs, doc_id, scores[doc_id], gamma)
    #         elif stemming:
    #             for doc_id, wdk in self.model.inverted_index_stemmed.get(term, []):
    #                 scores[doc_id] += wqk * wdk
    #                 self.model._insert_into_top_docs(top_docs, doc_id, scores[doc_id], gamma)
    #         else:
    #             for doc_id, wdk in self.model.inverted_index.get(term, []):
    #                 scores[doc_id] += wqk * wdk
    #                 self.model._insert_into_top_docs(top_docs, doc_id, scores[doc_id], gamma)
    #
    #         if self.model._can_terminate_early(top_docs, query_vector, sorted_query_terms, gamma):
    #             break
    #
    #         # Return the top gamma results
    #     return [(score, self.model.documents[doc_id]) for doc_id, score in
    #             sorted(top_docs.items(), key=lambda x: x[1], reverse=True)]

    def buckley_lewit_search(self, query: str, stemming: bool, stop_word_filtering: bool, gamma: int) -> list:
        if stop_word_filtering and stemming:
            if self.model.inverted_index_filtered_stemmed is None:
                self.model.inverted_index_filtered_stemmed = defaultdict(list)
                self.model.build_inv_list(self.collection, stop_word_filtering, stemming)
                self.model.calculate_idf(stopword_filtering=stop_word_filtering, stemming=stemming)
        elif stop_word_filtering:
            if self.model.inverted_index_filtered is None:
                self.model.inverted_index_filtered = defaultdict(list)
                self.model.build_inv_list(self.collection, stop_word_filtering, stemming)
                self.model.calculate_idf(stopword_filtering=stop_word_filtering, stemming=stemming)
        elif stemming:
            if self.model.inverted_index_stemmed is None:
                self.model.inverted_index_stemmed = defaultdict(list)
                self.model.build_inv_list(self.collection, stop_word_filtering, stemming)
                self.model.calculate_idf(stopword_filtering=stop_word_filtering, stemming=stemming)
        else:
            if self.model.inverted_index is None:
                self.model.inverted_index = defaultdict(list)
                self.model.build_inv_list(self.collection, stop_word_filtering, stemming)
                self.model.calculate_idf(stopword_filtering=stop_word_filtering, stemming=stemming)
        # query = stem_query_terms(query)
        query_vector = self.model.query_to_representation(query, stopword_filtering=stop_word_filtering, stemming=stemming)
        scores = defaultdict(float)
        top_docs = {}

        # Sort query terms by descending weights
        sorted_query_terms = sorted(query_vector.items(), key=lambda item: item[1], reverse=True)

        for term, wqk in sorted_query_terms:
            if wqk <= 0:
                continue
            if stop_word_filtering and stemming:
                for doc_id, wdk in self.model.inverted_index_filtered_stemmed.get(term, []):
                    scores[doc_id] += wqk * wdk
                    self.model._insert_into_top_docs(top_docs, doc_id, scores[doc_id], gamma)
            elif stop_word_filtering:
                for doc_id, wdk in self.model.inverted_index_filtered.get(term, []):
                    scores[doc_id] += wqk * wdk
                    self.model._insert_into_top_docs(top_docs, doc_id, scores[doc_id], gamma)
            elif stemming:
                for doc_id, wdk in self.model.inverted_index_stemmed.get(term, []):
                    scores[doc_id] += wqk * wdk
                    self.model._insert_into_top_docs(top_docs, doc_id, scores[doc_id], gamma)
            else:
                for doc_id, wdk in self.model.inverted_index.get(term, []):
                    scores[doc_id] += wqk * wdk
                    self.model._insert_into_top_docs(top_docs, doc_id, scores[doc_id], gamma)

            if self.model._can_terminate_early(top_docs, query_vector, sorted_query_terms, gamma):
                break

        # Return the top gamma results
        return [(score, self.model.documents[doc_id]) for doc_id, score in
                sorted(top_docs.items(), key=lambda x: x[1], reverse=True)]

    def signature_search(self, query: str, stemming: bool, stop_word_filtering: bool) -> list:
        """
        Fast Boolean query search using signatures for quicker processing.
        :param query: Query string
        :param stemming: Controls, whether stemming is used
        :param stop_word_filtering: Controls, whether stop-words are ignored in the search
        :return: List of tuples, where the first element is the relevance score and the second the corresponding
        document
        """
        self.model.build_term_signatures(self.collection, stop_word_filtering, stemming)
        query_signatures, term_groups = self.model.query_to_representation(query, stop_word_filtering, stemming)
        # print(query_signatures)
        # print(term_groups)

        results = []
        for document in self.collection:
            document_representation = self.model.document_to_representation(document, stop_word_filtering, stemming)

            if any(self.model.match(document_representation, query_sig) for query_sig in query_signatures):
                # print('halaaaaaaaaaaaaaaaaaaaaaa')
                # print(document.document_id)
                if self.model.string_matching(document, term_groups, stop_word_filtering, stemming):
                    results.append((1.0, document))

        return results

    # def parse_query(self,query: str) -> list:
    #     """
    #     Parses the query string into tokens considering Boolean operators.
    #     """
    #     # Define operator precedence
    #     precedence = {'-': 3, '&': 2, '|': 1}
    #     output = []
    #     operators = deque()
    #     query = query.lower()
    #     # Tokenize the query
    #     tokens = re.findall(r'\b\w+\b|[\-&|()]', query)
    #
    #     for token in tokens:
    #         if token.isalnum():
    #             output.append(token)
    #         elif token in precedence:
    #             while (operators and operators[-1] != '(' and
    #                    precedence.get(token, 0) <= precedence.get(operators[-1], 0)):
    #                 output.append(operators.pop())
    #             operators.append(token)
    #         elif token == '(':
    #             operators.append(token)
    #         elif token == ')':
    #             while operators and operators[-1] != '(':
    #                 output.append(operators.pop())
    #             operators.pop()  # Pop '('
    #
    #     while operators:
    #         output.append(operators.pop())
    #
    #     return output

    def parse_query(self,query: str) -> list:
        """
        Parses the query string into tokens considering Boolean operators.
        """
        # Define operator precedence
        precedence = {'-': 3, '&': 2, '|': 1}
        output = []
        operators = deque()
        query = query.lower()

        # Tokenize the query
        tokens = re.findall(r'\b\w+\b|[\-&|()]', query)

        for token in tokens:
            if token.isalnum():  # If the token is an operand (word)
                output.append(token)
                # Handle negation directly before an operand
                if operators and operators[-1] == '-':
                    output.append(operators.pop())
            elif token == '-':  # Handle negation
                operators.append(token)
            elif token in precedence:  # If the token is an operator
                while (operators and operators[-1] != '(' and
                       precedence.get(token, 0) <= precedence.get(operators[-1], 0)):
                    output.append(operators.pop())
                operators.append(token)
            elif token == '(':  # If the token is an open parenthesis
                operators.append(token)
            elif token == ')':  # If the token is a close parenthesis
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                operators.pop()  # Pop the open parenthesis
                # Check if negation applies to this sub-expression
                if operators and operators[-1] == '-':
                    output.append(operators.pop())

        while operators:
            output.append(operators.pop())

        return output

    # def parse_query(self,query: str) -> list:
    #     """
    #     Parses the query string into tokens considering Boolean operators.
    #     """
    #     # Define operator precedence and associativity (left for all)
    #     precedence = {'-': 3, '&': 2, '|': 1}
    #     output = []
    #     operators = deque()
    #     query = query.lower()
    #
    #     # Tokenize the query
    #     tokens = re.findall(r'\b\w+\b|[\-&|()]', query)
    #
    #     for token in tokens:
    #         if token.isalnum():  # If the token is an operand (word)
    #             output.append(token)
    #         elif token in precedence:  # If the token is an operator
    #             while (operators and operators[-1] != '(' and
    #                    precedence.get(token, 0) <= precedence.get(operators[-1], 0)):
    #                 output.append(operators.pop())
    #             operators.append(token)
    #         elif token == '(':  # If the token is an open parenthesis
    #             operators.append(token)
    #         elif token == ')':  # If the token is a close parenthesis
    #             while operators and operators[-1] != '(':
    #                 output.append(operators.pop())
    #             operators.pop()  # Pop the open parenthesis
    #             # Check if negation applies to this sub-expression
    #             if operators and operators[-1] == '-':
    #                 output.append(operators.pop())
    #
    #     while operators:
    #         output.append(operators.pop())
    #
    #     return output

    # def parse_query(self, query: str) -> list:
    #     """
    #     Parses the query string into tokens considering Boolean operators.
    #     """
    #     # Define operator precedence
    #     precedence = {'-': 3, '&': 2, '|': 1}
    #     output = []
    #     operators = deque()
    #     query = query.lower()
    #     # Tokenize the query
    #     tokens = re.findall(r'\b\w+\b|[\-&|()]', query)
    #
    #     for i, token in enumerate(tokens):
    #         if token.isalnum():
    #             output.append(token)
    #         # elif token == '-':
    #         #     operators.append(token)
    #         elif token in precedence:
    #             while (operators and operators[-1] != '(' and
    #                    precedence.get(token, 0) <= precedence.get(operators[-1], 0)):
    #                 output.append(operators.pop())
    #             operators.append(token)
    #         elif token == '(':
    #             operators.append(token)
    #         elif token == ')':
    #             while operators and operators[-1] != '(':
    #                 output.append(operators.pop())
    #             operators.pop()  # Pop '('
    #             # Check if negation applies to this sub-expression
    #             if operators and operators[-1] == '-':
    #                 output.append(operators.pop())
    #
    #     while operators:
    #         output.append(operators.pop())
    #
    #     return output

    # def parse_query(self, query: str) -> list:
    #     """
    #     Parses the query string into tokens considering Boolean operators.
    #     """
    #     # Define operator precedence
    #     precedence = {'-': 3, '&': 2, '|': 1}
    #     output = []
    #     operators = deque()
    #     query = query.lower()
    #     # Tokenize the query
    #     tokens = re.findall(r'\b\w+\b|[\-&|()]', query)
    #     # for term in tokens:
    #     #     if term == '&':
    #     #         # Process '&' operator: apply higher precedence operators before adding '&'
    #     #         # while operators and operators[-1] in '&|-':
    #     #         #     self.evaluate_query(output, operators.pop(),documents)
    #     #         operators.append(term)
    #     #     elif term == '|':
    #     #         # Process '|' operator: apply higher precedence '|' operators before adding '|'
    #     #         # while operators and operators[-1] == '|':
    #     #         #     self.apply_operator(output, operators.pop(),documents)
    #     #         operators.append(term)
    #     #     elif term == '-':
    #     #         operators.append(term) # Push '-' operator to the stack
    #     #     elif term == '(':
    #     #         operators.append(term) # Push '(' to the stack
    #     #     elif term == ')':
    #     #         # Process ')' operator: pop operators until matching '(' is found
    #     #         # while operators and operators[-1] != '(':
    #     #         #     self.apply_operator(output, operators.pop(),documents)
    #     #         operators.pop()  # remove the '('
    #     #     else:
    #     #         # print('term :',inv_list[term])
    #     #         # term_docs = inv_list.get(term, set()) # Get document IDs for the current term from inverted list
    #     #         # print('term_docs', term_docs)
    #     #         # if operators and operators[-1] == '-':
    #     #         #     # Handle negation '-' operator: remove documents from all available documents set
    #     #         #     operators.pop()
    #     #         #     all_docs = set(doc.document_id for doc in documents) # All document IDs in the collection
    #     #         #     term_docs = all_docs - term_docs # Calculate the negation of term_docs
    #     #         output.append(term)
    #
    #     for i, token in enumerate(tokens):
    #         if token.isalnum():
    #             output.append(token)
    #         elif token == '-':
    #             # Handle negation, especially when it applies to sub-expressions
    #             if i < len(tokens) - 1 and tokens[i + 1] == '(':
    #                 operators.append(token)
    #             else:
    #                 output.append(token)
    #         elif token in precedence:
    #             while (operators and operators[-1] != '(' and
    #                    precedence.get(token, 0) <= precedence.get(operators[-1], 0)):
    #                 output.append(operators.pop())
    #             operators.append(token)
    #         elif token == '(':
    #             if i > 0 and tokens[i - 1] == '-':
    #                 output.append(operators.pop())  # Ensure negation is applied directly
    #             operators.append(token)
    #         elif token == ')':
    #             while operators and operators[-1] != '(':
    #                 output.append(operators.pop())
    #             operators.pop()  # Pop '('
    #
    #             # Check if negation applies to this sub-expression
    #             if operators and operators[-1] == '-':
    #                 output.append(operators.pop())
    #
    #     while operators:
    #         output.append(operators.pop())
    #
    #     return output

    def evaluate_query(self,tokens: list, ground_truth: dict) -> set:
        stack = []
        for token in tokens:
            if token.isalnum():
                stack.append(ground_truth.get(token, set()))
            elif token == '&':
                right = stack.pop()
                left = stack.pop()
                stack.append(left & right)
            elif token == '|':
                right = stack.pop()
                left = stack.pop()
                stack.append(left | right)
            elif token == '-':
                operand = stack.pop()
                stack.append(ground_truth['ALL_DOCS'] - operand)
                # if len(stack) == 1:
                #     operand = stack.pop()
                #     stack.append(ground_truth['ALL_DOCS'] - operand)
                # else:
                #     right = stack.pop()
                #     left = stack.pop()
                #     stack.append(left - right)
        return stack[0] if stack else set()

    def calculate_precision(self,query: str, result_list: list[tuple]) -> float:
        ground_truth = self.read_ground_truth('raw_data/ground_truth.txt')
        ground_truth['ALL_DOCS'] = set(doc.document_id for score, doc in result_list)
        # print(ground_truth['ALL_DOCS'])
        tokens = self.parse_query(query)
        # print('tokens', tokens)
        terms = [term for term in tokens if term.isalnum()]
        # print(terms)
        if not all(term in ground_truth for term in terms):
            return -1
        relevant_ground_truth = self.evaluate_query(tokens, ground_truth)
        # print('relevant_ground_truth', relevant_ground_truth)
        relevant_docs = set(doc.document_id for score, doc in result_list if score > 0)
        if len(relevant_docs) == 0:
            return -1
        # print(len(relevant_docs & relevant_ground_truth))
        # print(len(relevant_docs))
        if not relevant_docs:
            return -1
        if not relevant_ground_truth:
            return -1  # Precision calculation not possible

        precision = len(relevant_docs & relevant_ground_truth) / len(relevant_docs)
        return precision

    def calculate_recall(self,query: str, result_list: list[tuple]) -> float:
        ground_truth = self.read_ground_truth('raw_data/ground_truth.txt')
        ground_truth['ALL_DOCS'] = set(doc.document_id for score, doc in result_list)
        query_terms = self.parse_query(query)
        terms = [term for term in query_terms if term.isalnum()]
        if not all(term in ground_truth for term in terms):
            return -1
        relevant_ground_truth = self.evaluate_query(query_terms, ground_truth)
        # print('relevant_ground_truth', relevant_ground_truth)
        relevant_docs = set(doc.document_id for score, doc in result_list if score > 0)
        if len(relevant_docs) == 0:
            return -1
        # print(len(relevant_docs & relevant_ground_truth))
        # print(len(relevant_ground_truth))
        if not relevant_docs:
            return -1
        if not relevant_ground_truth:
            return -1  # Precision calculation not possible
        recall = len(relevant_docs & relevant_ground_truth) / len(relevant_ground_truth)
        return recall

    # def calculate_precision(self,query: str, result_list: list[tuple]) -> float:
    #     # TODO: Implement this function (PR03)
    #     ground_truth = self.read_ground_truth('raw_data/ground_truth.txt')
    #     query_terms = re.findall(r'\b\w+\b', query)
    #     print(query_terms)
    #     # print(query_terms)
    #     terms = [term for term in query_terms if term.isalnum()]
    #     print(terms)
    #
    #     if not all(term in ground_truth for term in terms):
    #         return -1
    #     relevant_docs = set()
    #     relevant_docs = set(doc.document_id for score, doc in result_list if score == 1)
    #
    #     if len(relevant_docs) == 0:
    #         return -1
    #     # print(relevant_docs)
    #     # print(len(relevant_docs))
    #     # print(relevant_docs)
    #     relevant_ground_truth = set()
    #     for term in ground_truth.keys():
    #         if term in terms:
    #             relevant_ground_truth |= ground_truth[term]
    #             # print(relevant_ground_truth)
    #
    #     if not relevant_ground_truth:
    #         return -1  # Precision calculation not possible
    #
    #     precision = len(relevant_docs & relevant_ground_truth) / len(relevant_docs)
    #     return precision
    #
    #
    # def calculate_recall(self,query: str, result_list: list[tuple]) -> float:
    #     # TODO: Implement this function (PR03)
    #     ground_truth = self.read_ground_truth('raw_data/ground_truth.txt')
    #     query_terms2 = re.findall(r'\b\w+\b', query)
    #     terms = [term for term in query_terms2 if term.isalnum()]
    #     if not all(term in ground_truth for term in terms):
    #         return -1
    #     relevant_docs = set(doc.document_id for score, doc in result_list if score == 1)
    #     if len(relevant_docs) == 0:
    #         return -1
    #     relevant_ground_truth = set()
    #     for term in ground_truth.keys():
    #         if term in terms:
    #             relevant_ground_truth |= ground_truth[term]
    #
    #     if not relevant_ground_truth:
    #         return -1  # Recall calculation not possible
    #
    #     recall = len(relevant_docs & relevant_ground_truth) / len(relevant_ground_truth)
    #     return recall
        # if not result_list:
        #     return -1
        # query_terms2 = self.model.query_to_representation(query)
        # terms = [term for term in query_terms2 if term.isalnum()]
        # if not all(term in ground_truth for term in terms):
        #     return -1
        # relevant_docs = set(doc.document_id for _, doc in result_list)
        # relevant_ground_truth = set()
        # for term in ground_truth.keys():
        #     if term in terms:
        #         relevant_ground_truth |= ground_truth[term]
        #
        # if not relevant_ground_truth:
        #     return -1  # Recall calculation not possible
        #
        # recall = len(relevant_docs & relevant_ground_truth) / len(relevant_ground_truth)
        # return recall


if __name__ == '__main__':
    irs = InformationRetrievalSystem()
    irs.main_menu()
    exit(0)
