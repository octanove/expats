
from typing import Generic, List, TypeVar

import pandas as pd

from expats.data.instance import Instance, LabeledTextInstance, ScoredTextInstance


T = TypeVar("T", bound=Instance)


class Dataset(Generic[T]):
    """Basic Dataset class
    """
    def __init__(self, instances: List[T]):
        self.instances = instances

    def __getitem__(self, idx) -> T:
        return self.instances[idx]

    def __len__(self) -> int:
        return len(self.instances)

    def to_labeled(self) -> "Dataset[LabeledTextInstance]":
        _insts = [inst.to_labeled() for inst in self.instances]
        return Dataset(_insts)

    def to_scored(self) -> "Dataset[ScoredTextInstance]":
        _insts = [inst.to_scored() for inst in self.instances]
        return Dataset(_insts)

    def to_dataframe(self) -> pd.DataFrame:
        col_names = list(vars(self.instances[0]).keys())
        _dict = {key: [] for key in col_names}
        for inst in self.instances:
            _vars = vars(inst)
            for name in col_names:
                _dict[name].append(_vars[name])
        return pd.DataFrame(_dict)
