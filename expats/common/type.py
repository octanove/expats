
from enum import Enum


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


# profiler task types.
SingleTextInput = str
ClassificationOutput = str
RegressionOutput = float
