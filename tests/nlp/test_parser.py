
import pytest

from expats.nlp.parser import create_spacy_parser, sentence_tokenize_en


def test_create_spacy_parser():
    create_spacy_parser("en_core_web_sm")
    with pytest.raises(OSError):
        create_spacy_parser("not_found_parser")


@pytest.mark.parametrize(
    "text, expected_sents",
    [
        ("i am here. you are also here.", ["i am here.", "you are also here."]),
    ]
)
def test_sentence_tokenize_en(text, expected_sents):
    assert sentence_tokenize_en(text) == expected_sents
