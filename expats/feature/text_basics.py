
from typing import Dict

import numpy as np
import spacy

from expats.common.instantiate import BaseConfig
from expats.feature.base import Feature


@Feature.register
class NumberOfTokenFeature(Feature[spacy.tokens.doc.Doc]):
    def extract(self, _input):
        return np.array([len(_input)])


@Feature.register
class AverageTokenLengthFeature(Feature[spacy.tokens.doc.Doc]):
    def extract(self, _input):
        n_token = len(_input)
        return sum(len(token) for token in _input) / n_token


@Feature.register
class DocEmbeddingFeature(Feature[spacy.tokens.doc.Doc]):
    def extract(self, _input):
        return _input.vector


class UnigramLikelihoodFeatureConfig(BaseConfig):
    path: str
    sep: str = "\t"


@Feature.register
class UnigramLikelihoodFeature(Feature[spacy.tokens.doc.Doc]):
    config_class = UnigramLikelihoodFeatureConfig

    def __init__(self, word2freq: Dict[str, float]):
        self._word2freq = word2freq
        self._freq_sum = sum(word2freq.values())

    def extract(self, _input):
        # NOTE: smoothing by 1 to avoid zero devide error
        val = sum(np.log(self._word2freq.get(token.text, 1) / self._freq_sum) for token in _input)
        # NOTE: take average to remove length bias
        return np.array([val / len(_input)])

    @classmethod
    def create(cls, params: UnigramLikelihoodFeatureConfig):
        with open(params.path) as f:
            word2freq = {
                line.rstrip().split(params.sep)[0]: float(line.rstrip().split(params.sep)[1])
                for line in f
            }
        return cls(word2freq)
