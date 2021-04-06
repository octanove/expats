
from typing import Dict, List

from datasets import Dataset as HFDataset


def dict_to_hfdataset(dic: Dict[str, List]) -> HFDataset:
    """helper function to convert to huggingface's dataset class
    """
    return HFDataset.from_dict(dic)
