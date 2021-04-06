
import os
import pickle
from typing import Any, List, Dict, TypeVar, Union

import numpy as np
from sklearn.base import BaseEstimator
import spacy

from expats.common.instantiate import BaseConfig
from expats.common.type import ClassificationOutput, RegressionOutput
from expats.feature.base import Feature
from expats.ml.sklearn import create_ml_classifier, create_ml_regressor
from expats.nlp.parser import create_spacy_parser
from expats.profiler.base import TextProfiler, TextClassifier, TextRegressor


T = TypeVar("T", bound=Union[ClassificationOutput, RegressionOutput])
ESTIMATOR_FILENAME = "model.pkl"


class _Config(BaseConfig):
    features: Any  # FIXME: any to avoid: pydantic.error_wrappers.ValidationError: value is not a valid list
    classifier: Dict
    parser_name: str = "en_core_web_sm"


# FIXME: it's appropriate
class DocFeatureMLBase(TextProfiler[T]):
    config_class = _Config

    # FIXME: 'fit' method seems not to be defined in BaseEstimator interaface.
    def __init__(
        self,
        params: _Config,
        doc_features: List[Feature[spacy.tokens.doc.Doc]],
        estimator: BaseEstimator,
        parser: spacy.language.Language
    ):
        assert _is_valid_estimator(estimator), "estimator in args is not valid"
        super().__init__(params)
        self._doc_features = doc_features
        self._estimator = estimator
        self._parser = parser

    # FIXME: inefficient
    def _transform_batch(self, inputs: List[str]) -> np.ndarray:
        return np.vstack([self._transform(text) for text in inputs])

    def _transform(self, text: str) -> np.ndarray:  # return 1d array
        doc = self._parser(text)
        return np.hstack([
            feature.extract(doc) for feature in self._doc_features
        ])

    def _predict_batch_from_vector(self, xs: np.ndarray) -> np.ndarray:
        return self._estimator.predict(xs)

    def predict_batch(self, inputs: List[str]) -> List[T]:
        xs = self._transform_batch(inputs)
        ys = self._predict_batch_from_vector(xs)
        return ys.tolist()

    def fit(self, inputs: List[str], ys: List[T]):
        xs = self._transform_batch(inputs)
        self._estimator.fit(xs, ys)

    @classmethod
    def create(cls, params: _Config):
        doc_features = [
            Feature.create_from_factory(feat["type"], feat["params"])
            for feat in params.features
        ]
        estimator = cls._create_estimator(params.classifier)
        parser = create_spacy_parser(params.parser_name)
        return cls(
            params, doc_features, estimator, parser
        )

    @classmethod
    def _create_estimator(cls, estimator_config: Dict) -> BaseEstimator:
        raise NotImplementedError()

    @classmethod
    def _load_internal(cls, artifact_path: str, params: _Config) -> "DocFeatureMLBase":
        doc_features = [
            Feature.create_from_factory(feat["type"], feat["params"])
            for feat in params.features
        ]
        with open(_get_estimator_path(artifact_path), "rb") as fw:
            estimator = pickle.load(fw)
        parser = create_spacy_parser(params.parser_name)
        return cls(params, doc_features, estimator, parser)

    def _save_internal(self, artifact_path: str):
        with open(_get_estimator_path(artifact_path), "wb") as fw:
            pickle.dump(self._estimator, fw)


@TextClassifier.register
class DocFeatureMLClassifier(DocFeatureMLBase[ClassificationOutput]):
    @classmethod
    def _create_estimator(cls, estimator_config: Dict) -> BaseEstimator:
        return create_ml_classifier(
            estimator_config["type"],
            estimator_config["params"]
        )


@TextRegressor.register
class DocFeatureMLRegressor(DocFeatureMLBase[RegressionOutput]):
    @classmethod
    def _create_estimator(cls, estimator_config: Dict) -> BaseEstimator:
        return create_ml_regressor(
            estimator_config["type"],
            estimator_config["params"]
        )


def _is_valid_estimator(estimator: BaseEstimator):
    _dir = dir(estimator)
    if "fit" not in _dir:
        return False
    if "predict" not in _dir:
        return False
    return True


def _get_estimator_path(dir_path: str) -> str:
    return os.path.join(dir_path, ESTIMATOR_FILENAME)
