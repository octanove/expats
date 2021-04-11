
from typing import List

import numpy as np
import spacy
import pytest
from unittest import mock

from expats.feature.text_basics import (
    NumberOfTokenFeature,
    NumberOfTokenPerSentFeature,
    AverageTokenLengthFeature,
    UnigramLikelihoodFeature,
)


@pytest.mark.parametrize(
    "text, expected_value",
    [
        ("i am here", 3),
    ]
)
def test_number_of_token_feature(spacy_en, text, expected_value):
    doc = spacy_en(text)
    feature = NumberOfTokenFeature()
    np.testing.assert_array_equal(feature.extract(doc), np.array([expected_value]))


@pytest.mark.parametrize(
    "text, expected_value",
    [
        ("This is foo. That is foo bar.", (9 / 2)),
    ]
)
def test_number_of_token_per_sent_feature(spacy_en, text, expected_value):
    doc = spacy_en(text)
    feature = NumberOfTokenPerSentFeature()
    np.testing.assert_array_equal(feature.extract(doc), np.array([expected_value]))


@pytest.mark.parametrize(
    "text, expected_value",
    [
        ("i am here", 7 / 3),
        ("a ab b", 4 / 3)
    ]
)
def test_average_token_length_feature(spacy_en, text, expected_value):
    doc = spacy_en(text)
    feature = AverageTokenLengthFeature()
    np.testing.assert_array_equal(feature.extract(doc), np.array([expected_value]))


@pytest.mark.parametrize(
    "text, word2freq, expected_value",
    [
        ("i am", {"i": 4, "am": 3, "is": 2}, (np.log(4 / 9) + np.log(3 / 9)) / 2),
        ("i are", {"i": 4, "am": 3, "is": 2}, (np.log(4 / 9) + np.log(1 / 9)) / 2),  # NOTE: OOV case
    ]
)
def test_unigram_likelihood_feature(spacy_en, text, word2freq, expected_value):
    doc = spacy_en(text)
    feature = UnigramLikelihoodFeature(word2freq)
    np.testing.assert_array_equal(feature.extract(doc), np.array([expected_value]))
