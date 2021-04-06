
from abc import ABCMeta, abstractmethod
from typing import Dict, Generic, List, TypeVar

import numpy as np

from expats.common.instantiate import BaseConfig, ConfigFactoried
from expats.common.type import ClassificationOutput, RegressionOutput

T = TypeVar("T")
U = TypeVar("U")


class Converter(Generic[T, U], metaclass=ABCMeta):
    """Converting some data type into another one.
    This is used for converting classification task's output (e.g discrete label) into regression's one (e.g float value)
    """
    @abstractmethod
    def convert(self, inputs: List[T]) -> List[U]:
        raise NotImplementedError()


class ClassificationToRegression(Converter[ClassificationOutput, RegressionOutput], ConfigFactoried):
    pass


class RegressionToClassification(Converter[RegressionOutput, ClassificationOutput], ConfigFactoried):
    pass


@ClassificationToRegression.register
class PredifinedNumerizer(ClassificationToRegression):
    def __init__(self, mapper: Dict[str, RegressionOutput]):
        self._mapper = mapper

    def convert(self, inputs: List[ClassificationOutput]) -> List[RegressionOutput]:
        return [self._convert(input_) for input_ in inputs]

    def _convert(self, input_):
        return self._mapper[input_]


@ClassificationToRegression.register
class ToFloat(ClassificationToRegression):
    def convert(self, inputs: List[ClassificationOutput]) -> List[RegressionOutput]:
        return [float(input_) for input_ in inputs]


@RegressionToClassification.register
class RoundNearestInteger(RegressionToClassification):
    def convert(self, inputs: List[RegressionOutput]) -> List[ClassificationOutput]:
        return [str(int(np.rint(input_))) for input_ in inputs]


@RegressionToClassification.register
class MinMaxDenormalizedRoundNearestInteger(RegressionToClassification):
    class _Config(BaseConfig):
        x_min: float
        x_max: float

    config_class = _Config

    def __init__(self, x_min: float, x_max: float):
        self._x_min = x_min
        self._x_max = x_max

    def convert(self, inputs: List[RegressionOutput]) -> List[ClassificationOutput]:
        return [
            str(int(np.rint(_min_max_denormalization(input_, self._x_min, self._x_max))))
            for input_ in inputs
        ]


def _min_max_denormalization(x_norm: float, x_min: float, x_max: float) -> float:
    if (x_norm > 1) or (x_norm < 0) or (x_min > x_max):
        raise ValueError(
            f"Invalid setting to normalize: x_norm={x_norm}, x_min={x_min}, x_max={x_max}")
    return x_norm * (x_max - x_min) + x_min
