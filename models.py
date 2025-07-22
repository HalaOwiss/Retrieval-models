# Contains all retrieval models.

from abc import ABC, abstractmethod
from typing import List, Dict, Set
from pyparsing import Word, alphas, infixNotation, opAssoc, ParserElement, ParseException
import hashlib
from document import Document
import re
import math
from collections import defaultdict, Counter

class RetrievalModel(ABC):
    @abstractmethod
    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        """
        Converts a document into its model-specific representation.
        This is an abstract method and not meant to be edited. Implement it in the subclasses!
        :param document: Document object to be represented
        :param stopword_filtering: Controls, whether the document should first be freed of stopwords
        :param stemming: Controls, whether stemming is used on the document's terms
        :return: A representation of the document. Data type and content depend on the implemented model.
        """
        raise NotImplementedError()

    @abstractmethod
    def query_to_representation(self, query: str):
        """
        Determines the representation of a query according to the model's concept.
        :param query: Search query of the user
        :return: Query representation in whatever data type or format is required by the model.
        """
        raise NotImplementedError()

    @abstractmethod
    def match(self, document_representation, query_representation) -> float:
        """
        Matches the query and document presentation according to the model's concept.
        :param document_representation: Data that describes one document
        :param query_representation:  Data that describes a query
        :return: Numerical approximation of the similarity between the query and document representation. Higher is
        "more relevant", lower is "less relevant".
        """
        raise NotImplementedError()


class LinearBooleanModel(RetrievalModel):
    def __init__(self):
        super().__init__()
        # Enable packrat parsing for better performance
        ParserElement.enablePackrat()

        # Define the grammar for parsing the query string
        self.term = Word(alphas)
        self.expr = infixNotation(
            self.term,
            [
                ('-', 1, opAssoc.RIGHT),
                ('&', 2, opAssoc.LEFT),
                ('|', 2, opAssoc.LEFT),
                ('-', 2, opAssoc.LEFT),
            ]
        )
    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        if stopword_filtering == True and stemming == False:
            terms = document.filtered_terms
        elif stopword_filtering and stemming:
            terms = document.stemmed_filtered_terms
        elif stemming and stopword_filtering == False:
            terms = document.stemmed_terms
        else:
            terms = document.terms
        return set(terms)

    def query_to_representation(self, query: str):
        query = query.strip().lower()
        # Tokenize the query into words and operators
        tokens = re.findall(r'\w+|[&|()-]', query)

        # Reconstruct the query with '&' where necessary
        result = []
        for i in range(len(tokens)):
            result.append(tokens[i])
            # Check if the current token and the next token are both alphanumeric, add '&' between them
            if i + 1 < len(tokens):
                if tokens[i] not in '&|()-' and tokens[i + 1] not in '&|()-':
                    result.append('&')
        # Join the tokens back into a single string representation of the query
        return ''.join(result)

    def evaluate(self, parsed, terms):
        # Check if the parsed expression is a single term (base case)
        if isinstance(parsed, str):
            # print('parsed', parsed)
            return parsed in terms # Return True if the term is in the set of relevant terms
        if parsed[0] == '-':
            return not self.evaluate(parsed[1], terms) # Negate the evaluation of the operand
        # Handle binary operators '&' (AND) and '|' (OR)
        if parsed[1] == '&':
            # Evaluate the left and right operands recursively and perform AND operation
            return self.evaluate(parsed[0], terms) and self.evaluate(parsed[2], terms)
        if parsed[1] == '|':
            # Evaluate the left and right operands recursively and perform OR operation
            return self.evaluate(parsed[0], terms) or self.evaluate(parsed[2], terms)
        if parsed[1] == '-':
            # Evaluate the left and right operands recursively and perform OR operation
            return self.evaluate(parsed[0], terms) and not self.evaluate(parsed[2], terms)
        # Return False if the parsed expression does not match any expected structure
        return False

    def is_relevant(self, query, terms):
        # print(query)
        # print(terms)
        try:
            # Parse the query string using the defined grammar and obtain the parsed structure
            parsed_query = self.expr.parseString(query, parseAll=True)[0]
            # Evaluate the parsed query structure against the set of relevant terms
            return self.evaluate(parsed_query, terms)
        except ParseException as pe:
            print(f"Parse error: {pe}")
            return False

    def match(self, document_representation, query_representation) -> float:
        result = self.is_relevant(query_representation, document_representation)
        return 1.0 if result else 0.0

    def __str__(self):
        return 'Boolean Model (Linear)'


class InvertedListBooleanModel(RetrievalModel):
    def __init__(self):
        super().__init__()
        self.inverted_list = None
        self.inverted_list_filtered = None
        self.inverted_list_stemmed = None
        self.inverted_list_stemmed_filtered = None

    def build_inverted_list(self, documents: List[Document], stop_word_filtering: bool, stemming: bool) -> None:
        #Builds inverted lists based on the documents provided, depending on the specified filters.
        if stemming == False and stop_word_filtering == False:
            # Initialize inverted list for unstemmed and unfiltered terms
            self.inverted_list = {}
            for doc in documents:
                terms = self.document_to_representation(doc, stop_word_filtering, stemming)
                for term in terms:
                    if term not in self.inverted_list:
                        self.inverted_list[term] = set()
                    self.inverted_list[term].add(doc.document_id)

        if stemming == False and stop_word_filtering == True:
            # Initialize inverted list for unstemmed but filtered terms
            self.inverted_list_filtered = {}
            for doc in documents:
                terms = self.document_to_representation(doc, stop_word_filtering, stemming)
                for term in terms:
                    if term not in self.inverted_list_filtered:
                        self.inverted_list_filtered[term] = set()
                    self.inverted_list_filtered[term].add(doc.document_id)

        if stemming == True and stop_word_filtering == False:
            # Initialize inverted list for stemmed but unfiltered terms
            self.inverted_list_stemmed = {}
            for doc in documents:
                terms = self.document_to_representation(doc, stop_word_filtering, stemming)
                for term in terms:
                    if term not in self.inverted_list_stemmed:
                        self.inverted_list_stemmed[term] = set()
                    self.inverted_list_stemmed[term].add(doc.document_id)

        if stemming == True and stop_word_filtering == True:
            # Initialize inverted list for stemmed and filtered terms
            self.inverted_list_stemmed_filtered = {}
            for doc in documents:
                terms = self.document_to_representation(doc, stop_word_filtering, stemming)
                for term in terms:
                    if term not in self.inverted_list_stemmed_filtered:
                        self.inverted_list_stemmed_filtered[term] = set()
                    self.inverted_list_stemmed_filtered[term].add(doc.document_id)

    def document_to_representation(self, document: Document, stopword_filtering: bool = False,
                                   stemming: bool = False) -> Set[str]:
        if stopword_filtering == True and stemming == False:
            terms = document.filtered_terms
        elif stopword_filtering and stemming:
            terms = document.stemmed_filtered_terms
        elif stemming == True and stopword_filtering == False:
            terms = document.stemmed_terms
        else:
            terms = document.terms

        return set(terms)

    def query_to_representation(self, query: str) -> List[str]:
        # return re.findall(r'[\w\-\&\|\(\)]+', query)
        # return re.findall(r'\w+|[&|()-]', query)
        # Tokenize the query into words and operators
        query = query.lower()
        terms = re.findall(r'\w+|[&|()-]', query)
        # Reconstruct the query with '&' where necessary to ensure correct boolean logic
        result = []
        i = 0
        while i < len(terms):
            if terms[i].isalnum():
                result.append(terms[i])
                # Check if the next term is also alphanumeric and not an operator
                if i + 1 < len(terms) and terms[i + 1].isalnum():
                    result.append('&')
            else:
                result.append(terms[i])
            i += 1
        # print('Q:', result)
        return result

    def match(self, document_representation, query_representation) -> float:
        # Not used in inverted list search
        pass

    def evaluate_query(self, query_terms: List[str], inv_list,documents: List[Document]) -> Set[int]:
        stack = [] # Stack to hold sets of document IDs as we evaluate the query
        operators = [] # Stack to hold operators ('&', '|', '-', '(', ')')
        # print(inv_list)
        for term in query_terms:
            if term == '&':
                # Process '&' operator: apply higher precedence operators before adding '&'
                while operators and operators[-1] in '&|-':
                    self.apply_operator(stack, operators.pop(),documents)
                operators.append(term)
            elif term == '|':
                # Process '|' operator: apply higher precedence '|' operators before adding '|'
                while operators and operators[-1] == '|':
                    self.apply_operator(stack, operators.pop(),documents)
                operators.append(term)
            elif term == '-':
                operators.append(term) # Push '-' operator to the stack
            elif term == '(':
                operators.append(term) # Push '(' to the stack
            elif term == ')':
                # Process ')' operator: pop operators until matching '(' is found
                while operators and operators[-1] != '(':
                    self.apply_operator(stack, operators.pop(),documents)
                operators.pop()  # remove the '('
            else:
                # print('term :',inv_list[term])
                term_docs = inv_list.get(term, set()) # Get document IDs for the current term from inverted list
                # print('term_docs', term_docs)
                # if operators and operators[-1] == '-':
                #     # Handle negation '-' operator: remove documents from all available documents set
                #     operators.pop()
                #     all_docs = set(doc.document_id for doc in documents) # All document IDs in the collection
                #     term_docs = all_docs - term_docs # Calculate the negation of term_docs
                stack.append(term_docs)
                # print(f"Added term docs for '{term}': {term_docs}")

            # print(f"Operators stack: {operators}")
            # print(f"Documents stack: {stack}")
        # After processing all terms, apply any remaining operators on the stack
        while operators:
            self.apply_operator(stack, operators.pop(),documents)
        # Return the final set of document IDs that match the query
        return stack.pop() if stack else set()


    def apply_operator(self, stack: List[Set[int]], operator: str,documents: List[Document]) -> None:
        if operator == '&':
            # Apply intersection '&' operator: Pop top two sets from stack, compute intersection, push result back
            if len(stack) > 1:
                right = stack.pop()
                left = stack.pop()
                stack.append(left & right)
        elif operator == '|':
            # Apply union '|' operator: Pop top two sets from stack, compute union, push result back
            if len(stack) > 1:
                right = stack.pop()
                left = stack.pop()
                stack.append(left | right)
                # print('|: ', stack)
        elif operator == '-':
            # Apply negation '-' operator: Pop top set from stack, compute documents not in the set, push result back
            # if stack:
            #     top = stack.pop()
            #     all_docs = set(d.document_id for d in documents)
            #     stack.append(all_docs - top)
            # print(len(stack))
            if len(stack) == 1:
                top = stack.pop()
                all_docs = set(d.document_id for d in documents)
                stack.append(all_docs - top)
            else:
                right = stack.pop()
                left = stack.pop()
                stack.append(left - right)
                # print(left - right)
        # print(f"Applied operator '{operator}': {stack}")


# class SignatureBasedBooleanModel(RetrievalModel):
#     # TODO: Implement all abstract methods. (PR04)
#     def __init__(self):
#         raise NotImplementedError()  # TODO: Remove this line and implement the function.
#
#     def __str__(self):
#         return 'Boolean Model (Signatures)'

class SignatureBasedBooleanModel(RetrievalModel):
    def __init__(self):
        self.bit_vector_size = 64  # Signature length F
        self.num_bits = 7  # Signature weight m
        self.block_size = 4  # Overlay factor D
        self.documents = []
        self.term_dict = {}
        self.word_signatures = None
        self.word_signatures_filtered = None
        self.word_signatures_stemmed = None
        self.word_signatures_filtered_stemmed = None
        self.document_block_signatures = {}
        self.document_block_signatures_filtered = {}
        self.document_block_signatures_stemmed = {}
        self.document_block_signatures_filtered_stemmed = {}

    def __str__(self):
        return 'Boolean Model (Signatures)'

    def hash_term(self, term):
        bit_positions = set()
        seed = 0
        while len(bit_positions) < self.num_bits:
            hash_value = int(hashlib.sha256((term + str(seed)).encode()).hexdigest(), 16)
            bit_position = hash_value % self.bit_vector_size
            bit_positions.add(bit_position)
            seed += 1
        return bit_positions

    def create_word_signature(self, term):
        bit_positions = self.hash_term(term)
        signature = [0] * self.bit_vector_size
        for pos in bit_positions:
            signature[pos] = 1
        return signature

    def combine_signatures(self, signatures):
        combined = signatures[0][:]
        for sig in signatures[1:]:
            combined = [c | s for c, s in zip(combined, sig)]
        return combined

    def create_block_signatures(self, terms,term_sig):
        blocks = [terms[i:i + self.block_size] for i in range(0, len(terms), self.block_size)]
        # print("blocks")
        # print(blocks)
        block_signatures = []
        for block in blocks:
            block_sigs = [term_sig[term] for term in block if term in term_sig]
            if block_sigs:
                block_signature = self.combine_signatures(block_sigs)
                block_signatures.append(block_signature)
        return block_signatures

    def build_term_signatures(self, documents: List[Document], stop_word_filtering=False, stemming=False):
        if stemming == False and stop_word_filtering == False:
            # Initialize inverted list for unstemmed and unfiltered terms
            if self.word_signatures is None:
              self.word_signatures = {}
              for doc in documents:
                  terms = self.get_terms(doc, stop_word_filtering, stemming)
                  for term in terms:
                      if term not in self.word_signatures:
                          self.word_signatures[term] = self.create_word_signature(term)
                  # document_representation = self.create_block_signatures(terms)
                  block_signatures = self.create_block_signatures(terms,self.word_signatures)
                  self.document_block_signatures[doc.document_id] = block_signatures

        if stemming == False and stop_word_filtering == True:
            # Initialize inverted list for unstemmed but filtered terms
            if self.word_signatures_filtered is None:
                self.word_signatures_filtered = {}
                for doc in documents:
                    terms = self.get_terms(doc, stop_word_filtering, stemming)
                    for term in terms:
                        if term not in self.word_signatures_filtered:
                            self.word_signatures_filtered[term] = self.create_word_signature(term)
                    block_signatures = self.create_block_signatures(terms,self.word_signatures_filtered)
                    self.document_block_signatures_filtered[doc.document_id] = block_signatures

        if stemming == True and stop_word_filtering == False:
            # Initialize inverted list for stemmed but unfiltered terms
            if self.word_signatures_stemmed is None:
                self.word_signatures_stemmed = {}
                for doc in documents:
                    terms = self.get_terms(doc, stop_word_filtering, stemming)
                    for term in terms:
                        if term not in self.word_signatures_stemmed:
                            self.word_signatures_stemmed[term] = self.create_word_signature(term)
                    block_signatures = self.create_block_signatures(terms,self.word_signatures_stemmed)
                    self.document_block_signatures_stemmed[doc.document_id] = block_signatures

        if stemming == True and stop_word_filtering == True:
            # Initialize inverted list for stemmed and filtered terms
            if self.word_signatures_filtered_stemmed is None:
                self.word_signatures_filtered_stemmed = {}
                for doc in documents:
                    terms = self.get_terms(doc, stop_word_filtering, stemming)
                    for term in terms:
                        if term not in self.word_signatures_filtered_stemmed:
                            self.word_signatures_filtered_stemmed[term] = self.create_word_signature(term)
                    block_signatures = self.create_block_signatures(terms,self.word_signatures_filtered_stemmed)
                    self.document_block_signatures_filtered_stemmed[doc.document_id] = block_signatures


    def get_terms(self, document: Document, stopword_filtering=False, stemming=False):
        terms = document.terms
        if stopword_filtering:
            terms = document.filtered_terms
        if stemming:
            terms = document.stemmed_terms
        return terms
    # def query_to_representation(self, query: str):
    #     pass
    def document_to_representation(self, document: Document, stop_word_filtering=False, stemming=False):
        if stemming == False and stop_word_filtering == False:
            # if document.document_id in self.document_block_signatures:
                return self.document_block_signatures[document.document_id]
        elif stemming == False and stop_word_filtering == True:
            # if document.document_id in self.document_block_signatures_filtered:
                return self.document_block_signatures_filtered[document.document_id]
        elif stemming == True and stop_word_filtering == False:
                return self.document_block_signatures_stemmed[document.document_id]
        elif stemming == True and stop_word_filtering == True:
                return self.document_block_signatures_filtered_stemmed[document.document_id]


    def match(self, document_representation, query_representation) -> float:
        return any(self.compare_signatures(block_signature, query_representation) for block_signature in document_representation)

    def compare_signatures(self, block_signature, query_signature):
        # print(block_signature)
        # print(query_signature)
        # print(all(q <= b for q, b in zip(query_signature, block_signature)))
        return all(q <= b for q, b in zip(query_signature, block_signature))

    def string_matching(self, document, term_groups, stop_word_filtering=False, stemming=False):
        # Implement a detailed string matching logic here
        if stemming == False and stop_word_filtering == False:
                doc_terms = document.terms
        elif stemming == False and stop_word_filtering == True:
            doc_terms = document.filtered_terms
        elif stemming == True and stop_word_filtering == False:
                doc_terms = document.stemmed_terms
        elif stemming == True and stop_word_filtering == True:
                doc_terms = document.stemmed_filtered_terms
        # print(term_groups)
        # print(doc_terms)
        for group in term_groups:
          if all(term in doc_terms for term in group):
              return True
        return False


    def parse_query(self, query: str):
        # Split query using regex to preserve operators and parentheses
        query = query.lower()
        tokens = re.findall(r'\(|\)|&|\||\w+', query)

        # Clean up and filter tokens
        tokens = [token.strip() for token in tokens if token.strip()]
        # print('tokens :',tokens)
        result = []
        for i in range(len(tokens)):
            result.append(tokens[i])
            # Check if the current token and the next token are both alphanumeric, add '&' between them
            if i + 1 < len(tokens):
                if tokens[i] not in '&|()-' and tokens[i + 1] not in '&|()-':
                    result.append('&')
        # Join the tokens back into a single string representation of the query
        tokens = ''.join(result)
        tokens = re.findall(r'\(|\)|&|\||\w+', tokens)

        # Clean up and filter tokens
        # tokens = [token.strip() for token in tokens if token.strip()]
        # print('tokens :',tokens)
        stack = []
        current = []
        for token in tokens:
            if token == '(':
                stack.append(current)
                current = []
            elif token == ')':
                last = current
                current = stack.pop()
                current.append(last)
            elif token == '|':
                current.append('|')
            elif token == '&':
                current.append('&')
            else:
                current.append(token)

        return current

    def process_query_terms(self, query_terms):
        if isinstance(query_terms, str):
            return [[query_terms]]
        if '|' in query_terms:
            or_index = query_terms.index('|')
            left = query_terms[:or_index]
            right = query_terms[or_index + 1:]
            return self.process_query_terms(left) + self.process_query_terms(right)
        if '&' in query_terms:
            and_index = query_terms.index('&')
            left = query_terms[:and_index]
            right = query_terms[and_index + 1:]
            left_processed = self.process_query_terms(left)
            right_processed = self.process_query_terms(right)
            combined = []
            for l in left_processed:
                for r in right_processed:
                    combined.append(l + r)
            return combined
        if isinstance(query_terms, list):
            if len(query_terms) == 1:
                return self.process_query_terms(query_terms[0])
            combined = []
            for term in query_terms:
                processed = self.process_query_terms(term)
                for p in processed:
                    combined.append(p)
            return combined

    def query_to_representation(self, query: str, stop_word_filtering=False, stemming=False):
        parsed_query = self.parse_query(query)
        parsed_query = self.process_query_terms(parsed_query)
        query_signatures = []
        sig_test = {}
        if stemming == False and stop_word_filtering == False:
            sig_test = self.word_signatures
        elif stemming == False and stop_word_filtering == True:
            sig_test = self.word_signatures_filtered
        elif stemming == True and stop_word_filtering == False:
            sig_test = self.word_signatures_stemmed
        elif stemming == True and stop_word_filtering == True:
            sig_test = self.word_signatures_filtered_stemmed
        for term_group in parsed_query:
            # print(term_group)
            # print('------------')
            if isinstance(term_group, str):  # Single term case
                term_group = [term_group]
                # print('hiiii')
                # print(term_group)

            group_signature = None
            for term in term_group:
                # print('tem_group', term_group)
                term = term.lower()
                if term in sig_test:
                    if group_signature is None:
                        group_signature = sig_test[term][:]
                    else:
                        group_signature = [g | s for g, s in zip(group_signature, sig_test[term])]

            if group_signature:
                # print('group_signature', group_signature)
                query_signatures.append(group_signature)

        if query_signatures:
            # Union of all group signatures
            query_representation = query_signatures[0]
            # print('query_signatures[0]', query_signatures[0])
            for signature in query_signatures[1:]:
                query_representation = [q | s for q, s in zip(query_representation, signature)]
        else:
            query_representation = [0] * self.bit_vector_size

        return query_signatures, parsed_query

    def __str__(self):
        return 'Boolean Model (Signatures)'



# class VectorSpaceModel(RetrievalModel):
#     # TODO: Implement all abstract methods. (PR04)
#     def __init__(self):
#         raise NotImplementedError()  # TODO: Remove this line and implement the function.
#
#     def __str__(self):
#         return 'Vector Space Model'

# class VectorSpaceModel(RetrievalModel):
#     def __init__(self):
#         self.documents = []
#         self.inverted_index = None
#         self.inverted_index_filtered = None
#         self.inverted_index_stemmed = None
#         self.inverted_index_filtered_stemmed = None
#         self.doc_term_freqs = []
#         self.doc_vectors = []
#         self.idf = {}
#         self.idf_filtered = {}
#         self.idf_stemmed = {}
#         self.idf_filtered_stemmed = {}
#
#     def build_inv_list(self, documents: List[Document], stopword_filtering=False, stemming=False):
#         for document in documents:
#             if stopword_filtering and stemming:
#                 terms = document.stemmed_filtered_terms
#             elif stopword_filtering:
#                 terms = document.filtered_terms
#             elif stemming:
#                 terms = document.stemmed_terms
#             else:
#                 terms = document.terms
#             self.documents.append(document)
#             term_freq = Counter(terms)
#             # print("term_freq",term_freq)
#             self.doc_term_freqs.append(term_freq)
#             # print(self.doc_term_freqs)
#             for term, freq in term_freq.items():
#                 if stopword_filtering and stemming:
#                     self.inverted_index_filtered_stemmed[term].append((document.document_id, freq))
#                 elif stopword_filtering:
#                     self.inverted_index_filtered[term].append((document.document_id, freq))
#                 elif stemming:
#                     self.inverted_index_stemmed[term].append((document.document_id, freq))
#                 else:
#                     self.inverted_index[term].append((document.document_id, freq))
#                 # self.inverted_index[term].append((document.document_id, freq))
#
#     def calculate_idf(self, stopword_filtering=False, stemming=False):
#         N = len(self.documents)
#         if stopword_filtering and stemming:
#             for term, postings in self.inverted_index_filtered_stemmed.items():
#                 df = len(postings)  # num of the documents that contain the term
#                 self.idf_filtered_stemmed[term] = math.log(N / df)
#         elif stopword_filtering:
#             for term, postings in self.inverted_index_filtered.items():
#                 df = len(postings)  # num of the documents that contain the term
#                 self.idf_filtered[term] = math.log(N / df)
#         elif stemming:
#             for term, postings in self.inverted_index_stemmed.items():
#                 df = len(postings)  # num of the documents that contain the term
#                 self.idf_stemmed[term] = math.log(N / df)
#         else:
#             for term, postings in self.inverted_index.items():
#                 df = len(postings)  # num of the documents that contain the term
#                 self.idf[term] = math.log(N / df)
#
#     def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
#         term_freq = self.doc_term_freqs[document.document_id]
#         if stopword_filtering and stemming:
#             tf_idf_vector = {term: freq * self.idf_filtered_stemmed.get(term, 0) for term, freq in term_freq.items()}
#         elif stopword_filtering:
#             tf_idf_vector = {term: freq * self.idf_filtered.get(term, 0) for term, freq in term_freq.items()}
#         elif stemming:
#             tf_idf_vector = {term: freq * self.idf_stemmed.get(term, 0) for term, freq in term_freq.items()}
#         else:
#             tf_idf_vector = {term: freq * self.idf.get(term, 0) for term, freq in term_freq.items()}
#         return tf_idf_vector
#
#     def query_to_representation(self, query: str, stopword_filtering=False, stemming=False):
#         query_terms = query.lower()  # checkkkkkkkkkkkkkkkkkkkkkkkkkkk
#         # query_terms = query_terms.split()
#         query_terms = re.findall(r'\w+|[&|()-]', query_terms)
#         term_freq = Counter(query_terms)
#         if stopword_filtering and stemming:
#             query_vector = {term: freq * self.idf_filtered_stemmed.get(term, 0) for term, freq in term_freq.items()}
#         elif stopword_filtering:
#             query_vector = {term: freq * self.idf_filtered.get(term, 0) for term, freq in term_freq.items()}
#         elif stemming:
#             query_vector = {term: freq * self.idf_stemmed.get(term, 0) for term, freq in term_freq.items()}
#         else:
#             query_vector = {term: freq * self.idf.get(term, 0) for term, freq in term_freq.items()}
#         return query_vector
#
#     def match(self, document_representation, query_representation) -> float:
#         dot_product = sum(
#             document_representation.get(term, 0) * weight for term, weight in query_representation.items())
#         doc_norm = math.sqrt(sum(weight ** 2 for weight in document_representation.values()))
#         query_norm = math.sqrt(sum(weight ** 2 for weight in query_representation.values()))
#         if doc_norm == 0 or query_norm == 0:
#             return 0.0
#         return dot_product / (doc_norm * query_norm)
#
#     # def _insert_into_top_docs(self, top_docs, doc_score_pair, gamma):
#     #     top_docs.append(doc_score_pair)
#     #     top_docs.sort(key=lambda x: x[0], reverse=True)
#     #     if len(top_docs) > gamma + 1:
#     #         top_docs.pop()
#     #
#     # def _can_terminate_early(self, top_docs, query_vector, sorted_query_terms, gamma):
#     #     if len(top_docs) <= gamma:
#     #         return False
#     #
#     #     top_docs_sorted = sorted(top_docs, key=lambda x: x[0], reverse=True)
#     #     current_threshold = top_docs_sorted[gamma - 1][0]
#     #     print('current_threshold', current_threshold)
#     #     max_remaining_weight = sum(wqk for term, wqk in sorted_query_terms if wqk > 0)
#     #     print('max_remaining_weight', max_remaining_weight)
#     #     print('top_docs_sorted[gamma][0]', top_docs_sorted[gamma][0])
#     #     return top_docs_sorted[gamma][0] + max_remaining_weight <= current_threshold
#
#     def _insert_into_top_docs(self, top_docs, doc_id, score, gamma):
#         if doc_id in top_docs:
#             top_docs[doc_id] = max(top_docs[doc_id], score)
#         else:
#             top_docs[doc_id] = score
#         if len(top_docs) > gamma + 1:
#             lowest_doc_id = min(top_docs, key=top_docs.get)
#             del top_docs[lowest_doc_id]
#
#     def _can_terminate_early(self, top_docs, query_vector, sorted_query_terms, gamma):
#         if len(top_docs) < gamma:
#             return False
#
#         current_threshold = sorted(top_docs.values(), reverse=True)[gamma - 1]
#         max_remaining_weight = sum(wqk for term, wqk in sorted_query_terms if wqk > 0)
#         return list(sorted(top_docs.values(), reverse=True))[gamma] + max_remaining_weight <= current_threshold
class VectorSpaceModel(RetrievalModel):
    def __init__(self):
        self.documents = []
        self.inverted_index = None
        self.inverted_index_filtered = None
        self.inverted_index_stemmed = None
        self.inverted_index_filtered_stemmed = None
        self.doc_term_freqs = []
        self.doc_vectors = []
        self.idf = {}
        self.idf_filtered = {}
        self.idf_stemmed = {}
        self.idf_filtered_stemmed = {}

    def build_inv_list(self, documents: List[Document], stopword_filtering=False, stemming=False):
            for document in documents:
                if stopword_filtering and stemming:
                    terms = document.stemmed_filtered_terms
                elif stopword_filtering:
                    terms = document.filtered_terms
                elif stemming:
                    terms = document.stemmed_terms
                else:
                    terms = document.terms
                self.documents.append(document)
                term_freq = Counter(terms)
                self.doc_term_freqs.append(term_freq)
                for term, freq in term_freq.items():
                    if stopword_filtering and stemming:
                        self.inverted_index_filtered_stemmed[term].append((document.document_id, freq))
                    elif stopword_filtering:
                        self.inverted_index_filtered[term].append((document.document_id, freq))
                    elif stemming:
                        self.inverted_index_stemmed[term].append((document.document_id, freq))
                    else:
                        self.inverted_index[term].append((document.document_id, freq))

    def calculate_idf(self, stopword_filtering=False, stemming=False):
            N = len(self.documents)
            if stopword_filtering and stemming:
                for term, postings in self.inverted_index_filtered_stemmed.items():
                    df = len(postings)
                    self.idf_filtered_stemmed[term] = math.log(N / df)
            elif stopword_filtering:
                for term, postings in self.inverted_index_filtered.items():
                    df = len(postings)
                    self.idf_filtered[term] = math.log(N / df)
            elif stemming:
                for term, postings in self.inverted_index_stemmed.items():
                    df = len(postings)
                    self.idf_stemmed[term] = math.log(N / df)
            else:
                for term, postings in self.inverted_index.items():
                    df = len(postings)
                    self.idf[term] = math.log(N / df)

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
            term_freq = self.doc_term_freqs[document.document_id]
            if stopword_filtering and stemming:
                tf_idf_vector = {term: freq * self.idf_filtered_stemmed.get(term, 0) for term, freq in
                                 term_freq.items()}
            elif stopword_filtering:
                tf_idf_vector = {term: freq * self.idf_filtered.get(term, 0) for term, freq in term_freq.items()}
            elif stemming:
                tf_idf_vector = {term: freq * self.idf_stemmed.get(term, 0) for term, freq in term_freq.items()}
            else:
                tf_idf_vector = {term: freq * self.idf.get(term, 0) for term, freq in term_freq.items()}
            return tf_idf_vector

    def query_to_representation(self, query: str, stopword_filtering=False, stemming=False):
            query_terms = query.lower()
            query_terms = re.findall(r'\w+|[&|()-]', query_terms)
            term_freq = Counter(query_terms)
            if stopword_filtering and stemming:
                query_vector = {term: freq * self.idf_filtered_stemmed.get(term, 0) for term, freq in term_freq.items()}
            elif stopword_filtering:
                query_vector = {term: freq * self.idf_filtered.get(term, 0) for term, freq in term_freq.items()}
            elif stemming:
                query_vector = {term: freq * self.idf_stemmed.get(term, 0) for term, freq in term_freq.items()}
            else:
                query_vector = {term: freq * self.idf.get(term, 0) for term, freq in term_freq.items()}
            return query_vector

    def match(self, document_representation, query_representation) -> float:
            dot_product = sum(
                document_representation.get(term, 0) * weight for term, weight in query_representation.items())
            doc_norm = math.sqrt(sum(weight ** 2 for weight in document_representation.values()))
            query_norm = math.sqrt(sum(weight ** 2 for weight in query_representation.values()))
            if doc_norm == 0 or query_norm == 0:
                return 0.0
            return dot_product / (doc_norm * query_norm)

    def _insert_into_top_docs(self, top_docs, doc_id, score, gamma):
        if doc_id in top_docs:
            top_docs[doc_id] = max(top_docs[doc_id], score)
        else:
            top_docs[doc_id] = score
        if len(top_docs) > gamma + 1:
            lowest_doc_id = min(top_docs, key=top_docs.get)
            del top_docs[lowest_doc_id]

    def _can_terminate_early(self, top_docs, query_vector, sorted_query_terms, gamma):
        if len(top_docs) < gamma:
            return False

        current_threshold = sorted(top_docs.values(), reverse=True)[gamma - 1]
        max_remaining_weight = sum(wqk for term, wqk in sorted_query_terms if wqk > 0)
        sorted_scores = sorted(top_docs.values(), reverse=True)
        if len(sorted_scores) <= gamma:
            return sorted_scores[-1] + max_remaining_weight <= current_threshold
        return list(sorted(top_docs.values(), reverse=True))[gamma] + max_remaining_weight <= current_threshold

    def __str__(self):
        return 'Vector Space Model'


class FuzzySetModel(RetrievalModel):
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        raise NotImplementedError()  # TODO: Remove this line and implement the function.

    def __str__(self):
        return 'Fuzzy Set Model'
