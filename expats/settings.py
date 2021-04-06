
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings():
    spacy_parser_name: str = "en_core_web_sm"
    home_root_path: str = os.path.join(os.path.expanduser('~'), ".expats")
    is_debug: bool = bool(os.environ.get("IS_DEBUG")) or False
    random_seed: int = 46


SETTINGS = Settings()
