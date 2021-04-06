from typing import List

import numpy as np
import spacy
import pytest

from expats.feature.text_basics import (
    NumberOfTokenFeature,
    AverageTokenLengthFeature,
    UnigramLikelihoodFeature,
)


def _create_spacy_doc(words: List[str]) -> spacy.tokens.doc.Doc:
    return spacy.tokens.doc.Doc(spacy.vocab.Vocab(), words=words)


@pytest.mark.parametrize(
    "words, expected_value",
    [
        (["i", "am", "here"], 3),
    ]
)
def test_number_of_token_feature(words, expected_value):
    doc = _create_spacy_doc(words)
    feature = NumberOfTokenFeature()
    np.testing.assert_array_equal(feature.extract(doc), np.array([expected_value]))


@pytest.mark.parametrize(
    "words, expected_value",
    [
        (["i", "am", "here"], 7 / 3),
        (["a", "ab", "b"], 4 / 3)
    ]
)
def test_average_token_length_feature(words, expected_value):
    doc = _create_spacy_doc(words)
    feature = AverageTokenLengthFeature()
    np.testing.assert_array_equal(feature.extract(doc), np.array([expected_value]))


@pytest.mark.parametrize(
    "words, word2freq, expected_value",
    [
        (["i", "am"], {"i": 4, "am": 3, "is": 2}, (np.log(4 / 9) + np.log(3 / 9)) / 2),
        (["i", "are"], {"i": 4, "am": 3, "is": 2}, (np.log(4 / 9) + np.log(1 / 9)) / 2),  # NOTE: OOV case
    ]
)
def test_unigram_likelihood_feature(words, word2freq, expected_value):
    doc = _create_spacy_doc(words)
    feature = UnigramLikelihoodFeature(word2freq)
    np.testing.assert_array_equal(feature.extract(doc), np.array([expected_value]))
