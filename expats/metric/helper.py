
from typing import List, Optional

from scipy.stats import pearsonr as scipy_pearsonr
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def f1(gold_ys: List[int], pred_ys: List[int], average: Optional[str] = None) -> float:
    return f1_score(gold_ys, pred_ys, average=average)


def accuracy(gold_ys: List[int], pred_ys: List[int]) -> float:
    return accuracy_score(gold_ys, pred_ys)


def cohen_kappa(gold_ys: List[int], pred_ys: List[int], weights: Optional[str] = None) -> float:
    return cohen_kappa_score(gold_ys, pred_ys, weights=weights)


def pearsonr(gold_ys: List[float], pred_ys: List[float]) -> float:
    return scipy_pearsonr(gold_ys, pred_ys)
