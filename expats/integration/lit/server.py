
from typing import Any, Dict, List, Tuple, Union

from lit_nlp import dev_server
from lit_nlp import server_flags

from expats.common.type import TaskType
from expats.data.dataset import Dataset
from expats.integration.lit.dataset import TextClassificationLITDataset, TextRegressionLITDataset
from expats.integration.lit.model import LITModelForTextClassifier, LITModelForTextRegressor
from expats.profiler.base import TextProfiler
from expats.data.instance import LabeledTextInstance, ScoredTextInstance


def launch(
    task_type: TaskType,
    profiler: TextProfiler,
    dataset: Dataset[Union[LabeledTextInstance, ScoredTextInstance]]
):
    if task_type == TaskType.CLASSIFICATION:
        if not isinstance(dataset.instances[0], LabeledTextInstance):
            raise ValueError("Inconsistent type between Instance and TaskType")
        examples, labels = _setup_classification_dataset(dataset)
        models = {"text_classifier": LITModelForTextClassifier(profiler, labels)}
        lit_datasets = {"classification_dataset": TextClassificationLITDataset(examples, labels)}
        lit_demo = dev_server.Server(models, lit_datasets, **server_flags.get_flags())
        lit_demo.serve()
    elif task_type == TaskType.REGRESSION:
        if not isinstance(dataset.instances[0], ScoredTextInstance):
            raise ValueError("Inconsistent type between Instance and TaskType")
        examples = _setup_regression_dataset(dataset)
        models = {"text_regressor": LITModelForTextRegressor(profiler)}
        lit_datasets = {"regression_dataset": TextRegressionLITDataset(examples)}
        lit_demo = dev_server.Server(models, lit_datasets, **server_flags.get_flags())
        lit_demo.serve()
    else:
        raise ValueError(f"Unsupported task({task_type}) for launching LIT server")


def _setup_classification_dataset(dataset: Dataset[LabeledTextInstance]) -> Tuple[List[Dict[str, Any]], List[str]]:
    examples = [
        {"sentence": inst.text, "label": inst.label}
        for inst in dataset.instances
    ]
    labels = sorted(list(set([example["label"] for example in examples])))
    return (examples, labels)


def _setup_regression_dataset(dataset: Dataset[ScoredTextInstance]) -> List[Dict[str, Any]]:
    return [
        {"sentence": inst.text, "label": inst.score}
        for inst in dataset.instances
    ]
