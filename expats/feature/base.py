
import os
import pickle
from typing import Dict, Generic, List, Union, TypeVar

import numpy as np

from expats.common.serialization import Serializable
from expats.common.instantiate import BaseConfig, ConfigFactoried
from expats.error import InstantiationError, DeserializationError, SerializationError

T = TypeVar("T")


# TODO: how support stateful features (e.g TFIDF)
# TODO: consider compatibility for serialized object
class Feature(Generic[T], ConfigFactoried, Serializable):
    def fit(self, inputs: List[T]):
        # do nothing by default
        pass

    def extract(self, _input: T) -> np.ndarray:
        """
        Args:
            _input: source of feature
        Return:
            feature: extracted feature as 1d ndarray
        """
        raise NotImplementedError()

    @classmethod
    def create(cls, params: Union[BaseConfig, Dict]) -> "Feature":
        # construct with no args by default
        try:
            return cls()
        except Exception as e:
            raise InstantiationError(
                f"Fail to create {cls.__name__} instance. (reason: {str(e)}) Please override and implement appropriate 'create' method."
            )

    @classmethod
    def load(cls, artifact_path: str) -> "Feature":
        _path = os.path.join(artifact_path, f"{cls.__name__}.pkl")
        with open(_path, "rb") as f:
            try:
                obj = pickle.load(f)
            except Exception as e:
                raise DeserializationError(f"Fail to load pickled artifacts. (reason: {str(e)})")
        return obj

    def save(self, artifact_path: str):
        _path = os.path.join(artifact_path, f"{self.__class__.__name__}.pkl")
        with open(_path, "wb") as fw:
            try:
                pickle.dump(self, fw)
            except Exception as e:
                raise SerializationError(f"Fail to dump pickled artifacts. (reason: {str(e)})")
