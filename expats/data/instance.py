
from dataclasses import dataclass

from expats.common.type import ClassificationOutput, RegressionOutput


@dataclass(frozen=True)
class Instance():
    """Basic Instance class
    """

    def to_labeled(self) -> "LabeledTextInstance":
        raise NotImplementedError()

    def to_scored(self) -> "ScoredTextInstance":
        raise NotImplementedError()


@dataclass(frozen=True)
class SingleTextInstance(Instance):
    text: str


@dataclass(frozen=True)
class LabeledTextInstance(Instance):
    text: str
    label: ClassificationOutput

    def to_labeled(self):
        return self


@dataclass(frozen=True)
class ScoredTextInstance(Instance):
    text: str
    score: RegressionOutput

    def to_scored(self):
        return self
