
import os
from typing import List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from expats.common.instantiate import BaseConfig
from expats.settings import SETTINGS

SAVE_DIR = os.path.join(SETTINGS.home_root_path, "logs", "lightning_logs")


class PLTrainerConfig(BaseConfig):
    accumulate_grad_batches: int = 1
    gpus: Optional[Union[int, List[int]]]
    max_epochs: int
    min_epochs: int = 1


def create_pltrainer(config: PLTrainerConfig) -> pl.Trainer:
    logger = TensorBoardLogger(SAVE_DIR)
    return pl.Trainer(logger=logger, **config.to_dict())
