
from typing import Any, Dict, List

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types


class DatasetForLIT(lit_dataset.Dataset):
    def spec(self) -> Dict[str, lit_types.LitType]:
        raise NotImplementedError()


class TextClassificationLITDataset(lit_dataset.Dataset):
    def __init__(
        self,
        examples: List[Dict[str, Any]],
        labels: List[str]
    ):
        self._examples = examples
        self._labels = labels

    def spec(self):
        return {
            "sentence": lit_types.TextSegment(),
            "label": lit_types.CategoryLabel(vocab=self._labels)
        }


class TextRegressionLITDataset(lit_dataset.Dataset):
    def __init__(
        self,
        examples: List[Dict[str, Any]],
    ):
        self._examples = examples

    def spec(self):
        return {
            "sentence": lit_types.TextSegment(),
            "label": lit_types.RegressionScore(),
        }
