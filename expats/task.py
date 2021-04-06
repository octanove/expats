
import os
from typing import Dict, List, Tuple, Type, Union

import pandas as pd

from expats.common.config_util import dump_to_file, load_from_file
from expats.common.log import get_logger
from expats.common.type import TaskType
from expats.data.convert import ClassificationToRegression, RegressionToClassification
from expats.data.dataset import Dataset
from expats.data.factory import DatasetFactory
from expats.data.instance import LabeledTextInstance, ScoredTextInstance
from expats.metric.metric import Metric, ClassificationMetric, RegressionMetric
from expats.profiler.base import TextProfiler, TextClassifier, TextRegressor


TRAIN_CONFIG_FILENAME = "train_config.yaml"

logger = get_logger(__name__)


def train(config: Dict, artifact_path: str):
    task = TaskType(config["task"])

    dataset_config = config["dataset"]

    logger.info("Loading dataset ...")
    dataset = DatasetFactory.create_from_file(
        dataset_config["type"],
        dataset_config["params"]
    )
    dataset = _convert_dataset(task, dataset)
    logger.info(f"Dataset size: {len(dataset)}")

    xs, ys = extract_xys(task, dataset.to_dataframe())

    profiler_config = config["profiler"]

    logger.info("Initializing model ...")
    profiler_class = get_task_profiler_class(task)
    profiler = profiler_class.create_from_factory(profiler_config["type"], profiler_config["params"])

    logger.info("Start training")
    profiler.fit(xs, ys)

    logger.info("Saving ...")
    profiler.save(artifact_path)
    dump_to_file(config, os.path.join(artifact_path, TRAIN_CONFIG_FILENAME))


def predict(predict_config: Dict, output_file: str):
    logger.info("Loading artifacts ...")
    profiler, _ = load_artifacts(predict_config["artifact_path"])

    dataset_config = predict_config["dataset"]

    logger.info("Loading dataset ...")
    assert dataset_config["type"] == "line-by-line", "Only line-by-line dataset is available for predict"
    dataset = DatasetFactory.create_from_file(
        dataset_config["type"],
        dataset_config["params"]
    )
    logger.info(f"Dataset size: {len(dataset)}")

    xs = dataset.to_dataframe()["text"].to_list()

    logger.info("Making predictions ...")
    pred_ys = profiler.predict_batch(xs)

    logger.info("Writing prediction results to file ...")
    with open(output_file, "w") as fw:
        for x, y in zip(xs, pred_ys):
            print(f"{y}\t{x}", file=fw)


def evaluate(eval_config: Dict):
    logger.info("Loading artifacts ...")
    profiler, train_config = load_artifacts(eval_config["artifact_path"])
    train_task = TaskType(train_config["task"])

    dataset_config = eval_config["dataset"]

    logger.info("Loading dataset ...")
    dataset = DatasetFactory.create_from_file(
        dataset_config["type"],
        dataset_config["params"]
    )
    dataset = _convert_dataset(train_task, dataset)
    logger.info(f"Dataset size: {len(dataset)}")

    xs, gold_ys = extract_xys(train_task, dataset.to_dataframe())

    logger.info("Making predictions ...")
    pred_ys = profiler.predict_batch(xs)

    metrics_config = eval_config["metrics"]
    metric_report: Dict[str, float] = {}
    logger.info("Calcurating metrics ...")
    for _target_task in metrics_config.keys():
        _target_task_type = TaskType(_target_task)
        # convert model prediction/gold data for evaluation target task, not trainind task
        _gold_ys, _pred_ys = get_target_task_prediction(
            _target_task_type,
            train_task,
            gold_ys,
            pred_ys,
            eval_config["output_convert"]
        )

        for _metric_wise_config in metrics_config[_target_task]:
            task_metric_class = get_task_metric_class(_target_task_type)
            _metric = task_metric_class.create_from_factory(
                _metric_wise_config["type"],
                _metric_wise_config["params"]
            )
            metric_report[_metric.name] = _metric.calculate(
                [(g, p) for (g, p) in zip(_gold_ys, _pred_ys)]
            )
    logger.info(f"Results: {metric_report}")


def interpret(interpret_config: Dict):
    logger.info("Loading artifacts ...")
    profiler, train_config = load_artifacts(interpret_config["artifact_path"])
    train_task = TaskType(train_config["task"])

    dataset_config = interpret_config["dataset"]

    logger.info("Loading dataset")
    dataset = DatasetFactory.create_from_file(
        dataset_config["type"],
        dataset_config["params"]
    )
    dataset = _convert_dataset(train_task, dataset)
    logger.info(f"Dataset size: {len(dataset)}")

    # FIXME: better handling for integrations
    try:
        from expats.integration.lit.server import launch
        launch(
            train_task,
            profiler,
            dataset
        )
    except ImportError as e:
        logger.error(f"Failed to import. Please check if dependencies are properly installed. error={str(e)}")


def get_task_metric_class(
    task: TaskType
) -> Metric:
    if task == TaskType.CLASSIFICATION:
        return ClassificationMetric
    elif task == TaskType.REGRESSION:
        return RegressionMetric
    else:
        raise ValueError(f"Unsupported task({task}) for evaluation metrics")


def get_task_profiler_class(
    task: TaskType
) -> Type[TextProfiler]:
    if task == TaskType.CLASSIFICATION:
        return TextClassifier
    elif task == TaskType.REGRESSION:
        return TextRegressor
    else:
        raise ValueError(f"Unsupported task({task}) for extracting x and y")


def get_target_task_prediction(
    target_task,
    train_task,
    gold_ys,
    pred_ys,
    output_convert_config
):
    if target_task == train_task:
        return (gold_ys, pred_ys)
    elif (target_task == TaskType.REGRESSION) and (train_task == TaskType.CLASSIFICATION):
        # convert profiler prediction into Regression type
        converter_config = output_convert_config["classification_to_regression"]
        converter = ClassificationToRegression.create_from_factory(
            converter_config["type"],
            converter_config["params"]
        )
    elif (target_task == TaskType.CLASSIFICATION) and (train_task == TaskType.REGRESSION):
        converter_config = output_convert_config["regression_to_classification"]
        converter = RegressionToClassification.create_from_factory(
            converter_config["type"],
            converter_config["params"]
        )
    else:
        raise ValueError(f"Unexpected combinations for target task({target_task}) and train tasks({train_task})")

    return (converter.convert(gold_ys), converter.convert(pred_ys))


def load_artifacts(artifact_path: str) -> Tuple[TextProfiler, Dict]:
    train_config = load_from_file(os.path.join(artifact_path, TRAIN_CONFIG_FILENAME))
    train_task_type = TaskType(train_config["task"])
    profiler_type = train_config["profiler"]["type"]
    profiler_class = get_task_profiler_class(train_task_type)
    profiler = profiler_class.get_subclass(profiler_type).load(artifact_path)
    return (profiler, train_config)


def extract_xys(task: TaskType, dataset_df: pd.DataFrame) -> Tuple[List, List]:
    if task == TaskType.CLASSIFICATION:
        return (dataset_df["text"].tolist(), dataset_df["label"].tolist())
    elif task == TaskType.REGRESSION:
        return (dataset_df["text"].tolist(), dataset_df["score"].tolist())
    else:
        raise ValueError(f"Unsupported task({task}) for extracting x and y")


def _convert_dataset(task: TaskType, dataset: Dataset) -> Dataset[Union[LabeledTextInstance, ScoredTextInstance]]:
    if task == TaskType.CLASSIFICATION:
        return dataset.to_labeled()
    elif task == TaskType.REGRESSION:
        return dataset.to_scored()
    else:
        raise ValueError(f"Unsupported task({task}) for converting datset")
