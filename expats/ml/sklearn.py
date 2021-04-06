
from typing import Any, Dict

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor
)


def create_ml_classifier(_type: str, params: Dict[str, Any]):
    if _type == "rf":
        return RandomForestClassifier(**params)
    elif _type == "gb":
        return GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Invalid type: {_type}")


def create_ml_regressor(_type: str, params: Dict[str, Any]):
    if _type == "rf":
        return RandomForestRegressor(**params)
    elif _type == "gb":
        return GradientBoostingRegressor(**params)
    else:
        raise ValueError(f"Invalid type: {_type}")
