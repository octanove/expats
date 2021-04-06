
import csv
from dataclasses import dataclass
from glob import glob
import os
import sys

from expats.data.dataset import Dataset
from expats.data.instance import LabeledTextInstance
from expats.nlp.parser import sentence_tokenize_en


# based on Table1 in  https://www.mitpressjournals.org/doi/pdfplus/10.1162/tacl_a_00310
GRADE2SCALE = {
    "A1": 0,
    "A2": 20,
    "B1": 40,
    "B2": 60,
    "C1": 80,
    "C2": 100
}


@dataclass(frozen=True)
class CEFRGradedTextInstance(LabeledTextInstance):
    text: str
    label: str


def load_octanove_en_passages_data(
    passages_path: str,
    min_char_size: int = sys.maxsize,
    flatten_sentence: bool = True
) -> Dataset[CEFRGradedTextInstance]:
    """load english passage data whose instances are graded with CEFR

    Args:
        passages_path (str): path to corpus e.g) /path/to/education-data/en/passages/
        min_char_size (int, optional): filter out each instances by number of charactors to reduce noise.
        flatten_sentence (bool, optional): each passage will be splited into sentence unit to augment data

    Returns:
        Dataset[CEFRGradedTextInstance]: loaded dataset
    """
    dir2grade_mapper = {
        "a1": "A1",
        "a2": "A2",
        "b1": "B1",
        "b2": "B2",
        "c1": "C1",
        "c2": "C2",
    }

    instances = []
    for (dir_name, grade) in dir2grade_mapper.items():
        file_list = glob(os.path.join(passages_path, dir_name) + "/*")
        assert len(file_list) > 0, f"Missing files in dir {os.path.join(passages_path, dir_name)}"
        for file_path in file_list:
            with open(file_path) as f:
                text = f.read()
                if flatten_sentence is False:  # one text file as one instance
                    inst = CEFRGradedTextInstance(__clean_text(text), grade)
                    instances.append(inst)
                else:  # one each sentence in one text file as one instance
                    _insts = [
                        CEFRGradedTextInstance(sentence, grade)
                        for sentence in sentence_tokenize_en(__clean_text(text))
                    ]
                    instances.extend(_insts)

    # filter by token size
    instances = [
        inst for inst in instances
        if len(inst.text) >= min_char_size
    ]
    return Dataset(instances)


def __clean_text(text):
    return text.replace("\n", " ")


def load_cefr_annotated_tsv_data(file_path: str) -> Dataset[CEFRGradedTextInstance]:
    """Load a TSV file annotated with CEFR.
    We assume that each line of the file is of "[CEFR level] \t [text]"

    Args:
        file_path (str): path to the TSV file

    Returns:
        Dataset[CEFRGradedTextInstance]: loaded dataset
    """

    instances = []
    with open(file_path) as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        next(reader)   # skip header
        for row in reader:
            label, text = row
            if label.upper() not in GRADE2SCALE:
                raise ValueError(f"Invalid CEFR label: {label}")
            inst = CEFRGradedTextInstance(text, label.upper())
            instances.append(inst)
    return Dataset(instances)
