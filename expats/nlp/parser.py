
from functools import lru_cache
from typing import List

import spacy


@lru_cache(maxsize=10)
def create_spacy_parser(name: str) -> spacy.language.Language:
    return spacy.load(name)


def sentence_tokenize_en(text: str) -> List[str]:
    p = create_spacy_parser("en_core_web_sm")
    doc = p(text)
    return [sent.text for sent in doc.sents]
