from typing import List, Tuple

from gensim.corpora import Dictionary


def create_dictionary(documents: List[List[str]]):
    return Dictionary(documents)
