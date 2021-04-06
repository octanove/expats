from contextlib import contextmanager
import os
import requests
import shutil
import subprocess
import time
from typing import Optional

import pytest

from expats.task import train, evaluate, predict
from expats.common.config_util import load_from_file

TRAIN_ARTIFACT_PATH = "log/unittest"
PREDICT_OUTPUT_PATH = "log/unittest.output"
CONFIG_FIXTURE_DIR = "tests/fixtures/config/"


@contextmanager
def start_interpret_server(interpret_config_path: str):
    proc = subprocess.Popen([
        "poetry", "run", "expats", "interpret", f"{interpret_config_path}"
    ])
    try:
        yield proc
    finally:
        proc.kill()


@pytest.mark.parametrize(
    "train_yaml_filename, evaluate_yaml_filename, predict_yaml_filename, interpret_yaml_filename",
    [
        ("train.yaml", "evaluate.yaml", "predict.yaml", None),
        ("train_cefr.yaml", "evaluate_cefr.yaml", "predict.yaml", None),
        ("train_bert_classifier.yaml", "evaluate.yaml", "predict.yaml", "interpret.yaml"),
        ("train_bert_regressor.yaml", "evaluate.yaml", "predict.yaml", "interpret.yaml"),
        ("train_distilbert_regressor.yaml", "evaluate.yaml", "predict.yaml", "interpret.yaml")
    ]
)
def test_e2e_pipeline(
    train_yaml_filename: str,
    evaluate_yaml_filename: str,
    predict_yaml_filename: str,
    interpret_yaml_filename: Optional[str]
):
    # clean in advance
    if os.path.exists(TRAIN_ARTIFACT_PATH):
        shutil.rmtree(TRAIN_ARTIFACT_PATH)
        os.mkdir(TRAIN_ARTIFACT_PATH)  # FIXME
    if os.path.exists(PREDICT_OUTPUT_PATH):
        os.remove(PREDICT_OUTPUT_PATH)

    train_config_path = os.path.join(CONFIG_FIXTURE_DIR, train_yaml_filename)
    train_config = load_from_file(train_config_path)
    train(train_config, TRAIN_ARTIFACT_PATH)

    eval_config_path = os.path.join(CONFIG_FIXTURE_DIR, evaluate_yaml_filename)
    eval_config = load_from_file(eval_config_path)
    evaluate(eval_config)

    predict_config_path = os.path.join(CONFIG_FIXTURE_DIR, predict_yaml_filename)
    predict_config = load_from_file(predict_config_path)
    predict(predict_config, PREDICT_OUTPUT_PATH)

    if interpret_yaml_filename is not None:
        interpret_config_path = os.path.join(CONFIG_FIXTURE_DIR, interpret_yaml_filename)
        # FIXME: better test
        with start_interpret_server(interpret_config_path):
            max_retries = 10
            second_per_request = 10
            n_fail = 0
            for i in range(max_retries):
                time.sleep(second_per_request)
                try:
                    response = requests.get("http://localhost:5432")
                    break
                except requests.exceptions.ConnectionError:
                    n_fail += 1
            assert n_fail < max_retries, "Fail to connect"
            assert response.status_code == 200
