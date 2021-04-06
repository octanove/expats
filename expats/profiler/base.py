
import os
from typing import Any, Dict, Generic, List, TypeVar, Union

from expats.common.config_util import dump_to_file, load_from_file
from expats.common.instantiate import BaseConfig, ConfigFactoried
from expats.common.serialization import Serializable
from expats.common.type import SingleTextInput, ClassificationOutput, RegressionOutput
from expats.common.log import get_logger

T = TypeVar("T")
U = TypeVar("U")

logger = get_logger()

PARAMS_FILENAME = "profiler.yaml"


class ProfilerBase(Generic[T, U], ConfigFactoried, Serializable):
    """Profiler class to solve some tasks.
    """
    def __init__(self, params: Union[Dict, BaseConfig]):
        self._params = params

    def fit(self, inputs: List[T], ys: List[U]):
        # do nothing by default
        logger.info("This profiler does not do anything when training.")

    def predict(self, input_: T) -> U:
        return self.predict_batch([input_])[0]

    def predict_batch(self, inputs: List[T]) -> List[U]:
        raise NotImplementedError()

    def interprete_via_prediction(self, inputs: List[T], ys: List[U]) -> List[Dict[str, Any]]:
        # interpret models, based on acual prediction e.g) saliency map
        raise NotImplementedError(f"This profiler class {self.__class__.__name__} doesn't support predication-based interepretation")

    def interpred_via_internal(self) -> Dict[str, Any]:
        # interpret models, based ontraining results e.g) feature importance
        raise NotImplementedError(f"This profiler class {self.__class__.__name__} doesn't support internal interepretation")

    @classmethod
    def load(cls, artifact_path: str) -> "ProfilerBase":
        param_dict = load_from_file(_get_param_path(artifact_path))
        params = param_dict if cls.config_class is None else cls.config_class.from_dict(param_dict)
        return cls._load_internal(artifact_path, params)

    @classmethod
    def _load_internal(cls, artifact_path: str, params: Union[Dict, BaseConfig]) -> "ProfilerBase":
        raise NotImplementedError()

    def save(self, artifact_path: str):
        _param_dict = self._params if type(self._params) == dict else self._params.to_dict()
        dump_to_file(_param_dict, _get_param_path(artifact_path))
        self._save_internal(artifact_path)

    def _save_internal(self, artifact_path: str):
        raise NotImplementedError()


# This class is just for type annotation
class TextProfiler(ProfilerBase[SingleTextInput, U]):
    pass


class TextClassifier(TextProfiler[ClassificationOutput]):
    pass


class TextRegressor(TextProfiler[RegressionOutput]):
    pass


def _get_param_path(dir_path: str):
    return os.path.join(dir_path, PARAMS_FILENAME)
