
from abc import ABCMeta, abstractmethod
from typing import Generic, List, Tuple, TypeVar

from sklearn.preprocessing import LabelEncoder

from expats.common.instantiate import ConfigFactoried
from expats.metric.helper import f1, accuracy, cohen_kappa, pearsonr, spearmanr

T = TypeVar("T")
U = TypeVar("U")


class Metric(Generic[T], metaclass=ABCMeta):
    """Metric class for quantitative evaluation
    """
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def calculate(self, inputs: List[T]) -> float:
        raise NotImplementedError


class ClassificationMetric(ConfigFactoried, Metric[Tuple[str, str]]):
    def calculate(self, inputs: List[Tuple[str, str]]) -> float:
        gold_ys, pred_ys = _split(inputs)
        # str to int
        le = LabelEncoder()
        le.fit(gold_ys + pred_ys)
        return self._calculate(le.transform(gold_ys), le.transform(pred_ys))

    def _calculate(self, gold_ys: List[int], pred_ys: List[int]) -> float:
        raise NotImplementedError


@ClassificationMetric.register
class MacroF1(ClassificationMetric):
    def _calculate(self, gold_ys: List[int], pred_ys: List[int]) -> float:
        return f1(gold_ys, pred_ys, "macro")


@ClassificationMetric.register
class MicroF1(ClassificationMetric):
    def _calculate(self, gold_ys: List[int], pred_ys: List[int]) -> float:
        return f1(gold_ys, pred_ys, "micro")


@ClassificationMetric.register
class Accuracy(ClassificationMetric):
    def _calculate(self, gold_ys: List[int], pred_ys: List[int]) -> float:
        return accuracy(gold_ys, pred_ys)


@ClassificationMetric.register
class QuadraticWeightedKappa(ClassificationMetric):
    def _calculate(self, gold_ys: List[int], pred_ys: List[int]) -> float:
        return cohen_kappa(gold_ys, pred_ys, weights="quadratic")


class RegressionMetric(ConfigFactoried, Metric[Tuple[float, float]]):
    def calculate(self, inputs: List[Tuple[float, float]]) -> float:
        gold_ys, pred_ys = _split(inputs)
        return self._calculate(gold_ys, pred_ys)

    def _calculate(self, gold_ys: List[float], pred_ys: List[float]) -> float:
        raise NotImplementedError


@RegressionMetric.register
class PearsonCorrelation(RegressionMetric):
    def _calculate(self, gold_ys: List[float], pred_ys: List[float]) -> float:
        return pearsonr(gold_ys, pred_ys)[0]


@RegressionMetric.register
class SpearmanCorrelation(RegressionMetric):
    def _calculate(self, gold_ys: List[float], pred_ys: List[float]) -> float:
        return spearmanr(gold_ys, pred_ys)[0]


def _split(inputs: List[Tuple[U, U]]) -> Tuple[List[U], List[U]]:
    _firsts = [first for (first, _) in inputs]
    _seconds = [second for (_, second) in inputs]
    return (_firsts, _seconds)
