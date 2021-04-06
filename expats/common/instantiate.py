
from abc import ABCMeta
from collections import defaultdict
from typing import Dict, Optional, Union

from pydantic import BaseModel


class BaseConfig(BaseModel):
    """Basic configuration data class
    """
    @classmethod
    def from_dict(cls, dic: Dict) -> "BaseConfig":
        return cls(
            **dic
        )

    def to_dict(self) -> Dict:
        return self.dict()


class ConfigFactoried(metaclass=ABCMeta):
    """Basic class to be created via configuration data class.
    This is inspired by OpenKiwi
    """
    config_class: Optional[BaseConfig] = None
    _subclasses: Dict[str, Dict[str, "ConfigFactoried"]] = defaultdict(dict)

    @classmethod
    def register(cls, subcls):
        cls._subclasses[cls.__name__][subcls.__name__] = subcls
        return subcls

    @classmethod
    def get_subclass(cls, subcls_name: str):
        if cls.__name__ not in cls._subclasses:
            raise KeyError(
                f"{cls.__name__}'s subclass is not registered. Empty. Availables: f{dict(cls._subclasses)}"
            )
        cls2subcls = cls._subclasses[cls.__name__]
        subcls = cls2subcls.get(subcls_name, None)
        if not subcls:
            raise KeyError(
                f"{subcls_name} is not registered in {cls.__name__}. Here is the list: {list(cls2subcls.keys())}"
            )
        return subcls

    @classmethod
    def create_from_factory(cls, subcls_name: str, params: Optional[Dict]):
        _params = params if params is not None else {}
        subcls = cls.get_subclass(subcls_name)
        if subcls.config_class:
            return subcls.create(
                subcls.config_class.from_dict(_params)
            )
        else:
            return subcls.create(_params)

    @classmethod
    def create(cls, params: Union[BaseConfig, Dict]):
        """
        instantiate methods. if config_class is set, params should be BaseConfig else Dict
        """
        # default implement
        _params = params if type(params) == dict else params.to_dict()
        return cls(**_params)
