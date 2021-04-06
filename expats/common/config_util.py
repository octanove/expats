
from typing import Any, Dict, List

from omegaconf import OmegaConf
import yaml


def dump_to_file(dic: Dict[str, Any], path: str):
    """dump dict-type configuration to file

    Args:
        dic (Dict[str, Any]): dict-typed configuration
        path (str): path to file to dump
    """
    conf = OmegaConf.create(dic)
    OmegaConf.save(conf, path)


def load_from_file(path: str) -> Dict[str, Any]:
    """load configuration yaml file

    Args:
        path (str): path to file to be loaded

    Returns:
        Dict[str, Any]: loaded configurations
    """
    config = OmegaConf.load(path)
    return yaml.load(OmegaConf.to_yaml(config), Loader=yaml.FullLoader)


def merge_with_dotlist(dic: Dict[str, Any], dotlist: List[str]) -> Dict[str, Any]:
    """overwrite configuration values in the dictionary

    Args:
        doc (Dict[str, Any]): dict-typed configulations
        dotlist (bool): dotlist to overwrite config params

    Returns:
        Dict[str, Any]: overwitten configurations
    """
    config = OmegaConf.create(dic)
    config.merge_with_dotlist(dotlist)
    return yaml.load(OmegaConf.to_yaml(config), Loader=yaml.FullLoader)
