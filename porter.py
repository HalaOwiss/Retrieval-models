# Contains all functions related to the porter stemming algorithm.

from document import Document
import re
# hala

VOWELS = "aeiou"
CONSONANTS = "bcdfghjklmnpqrstvwxyz"


def is_consonant(word, i):
    """Check if the character at the index i in the word is a consonant."""
    if word[i] in VOWELS:
        return False
    if word[i] == 'y':
        return i == 0 or not is_consonant(word, i - 1)
    return True


def get_measure(term: str) -> int:
    """
    Returns the measure m of a given term [C](VC){m}[V].
    :param term: Given term/word
    :return: Measure value m
    """
    m = 0
    vc_sequence = False
    for i in range(len(term)):
        if is_consonant(term, i):
            if vc_sequence:
                m += 1
                vc_sequence = False
        else:
            vc_sequence = True
    return m


def condition_v(stem: str) -> bool:
    """
    Returns whether condition *v* is true for a given stem (= the stem contains a vowel).
    :param stem: Word stem to check
    :return: True if the condition *v* holds
    """
    for i in range(len(stem)):
        if stem[i] in VOWELS:
            return True
        if stem[i] == 'y' and i > 0 and is_consonant(stem, i - 1):
            return True
    return False


def condition_d(stem: str) -> bool:
    """
    Returns whether condition *d is true for a given stem (= the stem ends with a double consonant (e.g. -TT, -SS)).
    :param stem: Word stem to check
    :return: True if the condition *d holds
    """
    if len(stem) > 1 and stem[-1] == stem[-2]:
        if is_consonant(stem, len(stem) - 1):
            return True
    return False


def cond_o(stem: str) -> bool:
    """
    Returns whether condition *o is true for a given stem (= the stem ends cvc, where the second c is not W, X or Y
    (e.g. -WIL, -HOP)).
    :param stem: Word stem to check
    :return: True if the condition *o holds
    """
    if len(stem) >= 3 and is_consonant(stem, len(stem) - 3) and stem[-2] in VOWELS:
        if stem[-1] not in 'wxy' and is_consonant(stem, len(stem) - 1):
            return True
    return False


def stem_term(term: str) -> str:
    """
    Stems a given term of the English language using the Porter stemming algorithm.
    :param term:
    :return:
    """
    if term.endswith("sses"):
        term = term[:-2]
    elif term.endswith("ies"):
        term = term[:-2]
    elif term.endswith("ss"):
        pass
    elif term.endswith("s"):
        term = term[:-1]

    # Step 1b
    if term.endswith("eed"):
        if get_measure(term[:-3]) > 0:
            term = term[:-1]
    elif term.endswith("ed"):
        if condition_v(term[:-2]):
            term = term[:-2]
            if term.endswith(("at", "bl", "iz")):
                term += "e"
            elif condition_d(term) and term[-1] not in 'lsz':  # check if true
                term = term[:-1]
            elif cond_o(term) and get_measure(term) == 1:
                term += "e"
    elif term.endswith("ing"):
        if condition_v(term[:-3]):
            term = term[:-3]
            if term.endswith(("at", "bl", "iz")):
                term += "e"
            elif condition_d(term) and term[-1] not in 'lsz':  # check if true
                term = term[:-1]
            elif cond_o(term) and get_measure(term) == 1:
                term += "e"

    # Step 1c
    if term.endswith("y") and condition_v(term[:-1]):
        term = term[:-1] + "i"
    # Step 2
    pair_tests = [('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'), ('anci', 'ance'), ('izer', 'ize'),
                  ('abli', 'able'), ('alli', 'al'), ('entli', 'ent'), ('eli', 'e'), ('ousli', 'ous'),
                  ('ization', 'ize'),
                  ('ation', 'ate'), ('ator', 'ate'), ('alism', 'al'), ('iveness', 'ive'), ('fulness', 'ful'),
                  ('ousness', 'ous'), ('aliti', 'al'), ('iviti', 'ive'), ('biliti', 'ble'),('xflurti','xti')]
    for stem, subs in pair_tests:
        if term.endswith(stem):
            if get_measure(term[:-len(stem)]) > 0:
                term = term[:-len(stem)] + subs

    # Step 3
    pair_tests2 = [('icate', 'ic'), ('ative', ''), ('alize', 'al'), ('iciti', 'ic'), ('ical', 'ic'), ('ful', ''),
                   ('ness', '')]
    for stem, subs in pair_tests2:
        if term.endswith(stem):
            if get_measure(term[:-len(stem)]) > 0:
                term = term[:-len(stem)] + subs

    suffixes = ['al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant', 'ement', 'ment', 'ent', 'ou', 'ism', 'ate',
                'iti', 'ous', 'ive', 'ize']
    for suffix in suffixes:
        if term.endswith(suffix):
            if get_measure(term[:-len(suffix)]) > 1:
                term = term[:-len(suffix)]

    if get_measure(term[:-3]) > 1:
        if term.endswith('ion'):
            temp = term[:-3]
            if temp.endswith(('s', 't')):
                term = temp
    # Step 5a
    if term.endswith("e"):
        if get_measure(term[:-1]) > 1:
            term = term[:-1]
        elif get_measure(term[:-1]) == 1 and not cond_o(term[:-1]):
            term = term[:-1]

    # Step 5b
    if term.endswith("ll") and get_measure(term[:-1]) > 1:
        term = term[:-1]
    return term


def stem_all_documents(collection: list[Document]):
    """
    For each document in the given collection, this method uses the stem_term() function on all terms in its term list.
    Warning: The result is NOT saved in the document's term list, but in the extra field stemmed_terms!
    :param collection: Document collection to process
    """
    for doc in collection:
        doc.stemmed_terms = [stem_term(term) for term in doc.terms]
        if doc.filtered_terms:
            doc.stemmed_filtered_terms = [stem_term(term) for term in doc.filtered_terms]


def stem_query_terms(query: str) -> str:
    """
    Stems all terms in the provided query string.
    :param query: User query, may contain Boolean operators and spaces.
    :return: Query with stemmed terms
    """
    # terms = query.split()
    terms = re.findall(r'\w+|[&|()-]', query)
    stemmed_terms = [stem_term(term) for term in terms]
    return ' '.join(stemmed_terms)

