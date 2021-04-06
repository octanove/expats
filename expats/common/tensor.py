
from typing import Dict, List

import numpy as np


def _extract_batch_length(preds: Dict[str, np.ndarray]) -> int:
    """Extracts batch length of predictions."""
    batch_length = None
    for key, value in preds.items():
        batch_length = batch_length or value.shape[0]
        if value.shape[0] != batch_length:
            raise ValueError(f"Batch length of predictions should be same. {key} has different batch length than others.")
    return batch_length


def unbatch_preds(preds: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
    """Unbatch predictions, as in estimator.predict().
    """
    return [{key: value[i] for key, value in preds.items()} for i in range(_extract_batch_length(preds))]
