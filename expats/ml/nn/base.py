
from collections import OrderedDict
from typing import Dict

import torch
import pytorch_lightning as pl

import expats
from expats.common.instantiate import ConfigFactoried
from expats.common.serialization import Serializable


class NNModuleBase(pl.LightningModule, Serializable, ConfigFactoried):
    """
    Basic class for implementing Neural Network-based methods.
    inspired by OpenKiwi.
    """
    def __init__(self, params):
        super().__init__()
        self._params = params

    def forward(self, **kwargs):
        raise NotImplementedError()

    def training_step(self, **kwargs):
        raise NotImplementedError()

    def configure_optimizers(self):
        raise NotImplementedError()

    def validation_step(self, **kwargs):
        pass

    def forward_for_interpretation(self, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(f"This Neural Network class {self.__class__.__name__} doesn't support interepretation")

    @classmethod
    def load(cls, artifact_path: str) -> "NNModuleBase":
        # NOTE: load pretrained weights. (basically not for checkpoint purposes)
        module_dict = torch.load(artifact_path)
        nn_module = cls.create(module_dict["params"])
        nn_module.load_state_dict(module_dict["state_dict"])
        nn_module.eval()
        return nn_module

    def save(self, artifact_path: str):
        # NOTE: load pretrained weights. (basically not for checkpoint purposes)
        module_dict = OrderedDict(
            {
                "expats.__version__": expats.__version__,
                "torch.__version__": torch.__version__,
                "pl.__version__": pl.__version__,
                "state_dict": self.state_dict(),
                "params": self._params.to_dict()
            }
        )
        torch.save(module_dict, artifact_path)

    @classmethod
    def create(self, params: Dict) -> "NNModuleBase":
        """create model with initialized parameters
        """
        raise NotImplementedError()
