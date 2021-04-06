
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from expats.common.log import get_logger
from expats.data.dataset import Dataset
from expats.data.instance import Instance, LabeledTextInstance, ScoredTextInstance


logger = get_logger(__name__)

# NOTE: different from https://github.com/nusnlp/nea/blob/becd233ccd4788fd307da77a41b4731d31c0fab9/nea/asap_reader.py#L14
PROMPT_ID2SCORE_RANGE = {
    -1: (0, 60),
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (2, 24),
    8: (10, 60),
}


@dataclass(frozen=True)
class ASAPAESInstance(Instance):
    """Instance for asap-aes dataset
    training tsv information is here
        RangeIndex: 12976 entries, 0 to 12975
        Data columns (total 28 columns):
        #   Column          Non-Null Count  Dtype
        ---  ------          --------------  -----
        0   essay_id        12976 non-null  int64
        1   essay_set       12976 non-null  int64
        2   essay           12976 non-null  object
        3   rater1_domain1  12976 non-null  int64
        4   rater2_domain1  12976 non-null  int64
        5   rater3_domain1  128 non-null    float64
        6   domain1_score   12976 non-null  int64
        7   rater1_domain2  1800 non-null   float64
        8   rater2_domain2  1800 non-null   float64
        9   domain2_score   1800 non-null   float64
        10  rater1_trait1   2292 non-null   float64
        11  rater1_trait2   2292 non-null   float64
        12  rater1_trait3   2292 non-null   float64
        13  rater1_trait4   2292 non-null   float64
        14  rater1_trait5   723 non-null    float64
        15  rater1_trait6   723 non-null    float64
        16  rater2_trait1   2292 non-null   float64
        17  rater2_trait2   2292 non-null   float64
        18  rater2_trait3   2292 non-null   float64
        19  rater2_trait4   2292 non-null   float64
        20  rater2_trait5   723 non-null    float64
        21  rater2_trait6   723 non-null    float64
        22  rater3_trait1   128 non-null    float64
        23  rater3_trait2   128 non-null    float64
        24  rater3_trait3   128 non-null    float64
        25  rater3_trait4   128 non-null    float64
        26  rater3_trait5   128 non-null    float64
        27  rater3_trait6   128 non-null    float64
        dtypes: float64(22), int64(5), object(1)

    And also attached some fields.
        - domain1_score_normalized: normalized score of domain1_score
    """
    essay_id: int
    essay_set: int
    essay: str
    rater1_domain1: float
    rater2_domain1: float
    rater3_domain1: Optional[float]
    domain1_score: float
    rater1_domain2: Optional[float]
    domain2_score: Optional[float]
    domain1_score_normalized: float
    # FIXME: support more field e.g) rater1_trait1

    def to_labeled(self):
        return LabeledTextInstance(
            text=self.essay,
            label=str(self.domain1_score)
        )

    def to_scored(self):
        return ScoredTextInstance(
            text=self.essay,
            score=self.domain1_score_normalized
        )


def load_asap_aes(path: str, prompt_id: int = -1) -> Dataset[ASAPAESInstance]:
    _df = pd.read_csv(path, sep="\t", encoding="ISO-8859-1")
    if prompt_id not in PROMPT_ID2SCORE_RANGE:
        raise ValueError(f"Unavailable prompt id {prompt_id}")
    if prompt_id != -1:
        _df = _df[_df["essay_set"] == prompt_id]
    _min, _max = PROMPT_ID2SCORE_RANGE[prompt_id]
    _df["domain1_score_normalized"] = _df["domain1_score"].apply(
        lambda x: _assign_normalized_score(x, _min, _max)
    )
    instances = [
        ASAPAESInstance(
            essay_id=record["essay_id"],
            essay_set=record["essay_set"],
            essay=record["essay"],
            rater1_domain1=record["rater1_domain1"],
            rater2_domain1=record["rater2_domain1"],
            rater3_domain1=record.get("rater3_domain1", None),
            domain1_score=record["domain1_score"],
            rater1_domain2=record.get("rater1_domain2", None),
            domain2_score=record.get("domain2_score", None),
            domain1_score_normalized=record["domain1_score_normalized"]
        ) for record in _df.to_dict(orient='records')
    ]
    return Dataset(instances)


def _assign_normalized_score(x: float, x_min: float, x_max: float) -> float:
    try:
        return _min_max_normalization(x, x_min, x_max)
    except ValueError as e:
        logger.warning(f"Fail to normalize. Force assign: {e}.")
        return 0.0 if x < x_min else 1.0


def _min_max_normalization(x: float, x_min: float, x_max: float) -> float:
    if (x < x_min) or (x > x_max):
        raise ValueError(f"Invalid setting to normalize: x={x}, x_min={x_min}, x_max={x_max}")
    return (x - x_min) / (x_max - x_min)
