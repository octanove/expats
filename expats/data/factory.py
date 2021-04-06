
from typing import Any, Dict

from expats.data.asap_aes import load_asap_aes
from expats.data.dataset import Dataset
from expats.data.cefr import load_octanove_en_passages_data, load_cefr_annotated_tsv_data
from expats.data.instance import SingleTextInstance


class DatasetFactory():
    """creating dataset with some settings
    """
    @classmethod
    def create_from_file(cls, _type: str, params: Dict[str, Any]) -> Dataset:
        if _type == "octanove-cefr-en-passage":
            return load_octanove_en_passages_data(**params)
        elif _type == "cefr-tsv":
            return load_cefr_annotated_tsv_data(**params)
        elif _type == "line-by-line":
            return load_line_by_line(**params)
        elif _type == "asap-aes":
            return load_asap_aes(**params)
        else:
            raise ValueError(f"Invalid dataset type: {_type}")


def load_line_by_line(file_path: str) -> Dataset[SingleTextInstance]:
    """loading single text instances. one text instance per line.
    """
    with open(file_path, "r") as f:
        return Dataset([SingleTextInstance(text=line.rstrip()) for line in f])
