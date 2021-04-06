
BASE_DIR := $(shell pwd)
POETRY_RUN := poetry run

TRAIN_ARTIFACTS_DIR := ${BASE_DIR}/log/train_$(shell date +'%Y%m%d')
TRAIN_CONFIG_PATH := ${BASE_DIR}/config/train.yaml
EVALUATION_CONFIG_PATH := ${BASE_DIR}/config/evaluation.yaml
PREDICT_CONFIG_PATH := ${BASE_DIR}/config/predict.yaml
INTREPRET_CONFIG_PATH := ${BASE_DIR}/config/interpret.yaml

# override configs (defaults No overrides)
OVERRIDES := 

TENSORBORD_LOG_DIR := ...

install:
	poetry install

notebook:
	${POETRY_RUN} jupyter-notebook

train:
	@echo training
	mkdir -p ${TRAIN_ARTIFACTS_DIR}
	${POETRY_RUN} expats train ${TRAIN_CONFIG_PATH} ${TRAIN_ARTIFACTS_DIR} --overrides ${OVERRIDES}

train-debug:
	IS_DEBUG=true make train TRAIN_CONFIG_PATH=config/train_debug.yaml TRAIN_ARTIFACTS_DIR=log/debug

evaluate:
	@echo evaluation on pre-trained model
	${POETRY_RUN} expats evaluate ${EVALUATION_CONFIG_PATH} --overrides ${OVERRIDES}

evaluate-debug:
	IS_DEBUG=true make evaluate EVALUATION_CONFIG_PATH=config/evaluate_debug.yaml

predict:
	@echo evaluation on pre-trained model
	${POETRY_RUN} expats predict ${PREDICT_CONFIG_PATH} ${PREDICT_OUTPUT_PATH} --overrides ${OVERRIDES}

predict-debug:
	IS_DEBUG=true make predict PREDICT_CONFIG_PATH=config/predict_debug.yaml PREDICT_OUTPUT_PATH=log/debug_predict

interpret:
	@echo interpreting pre-trained model
	${POETRY_RUN} expats interpret ${INTREPRET_CONFIG_PATH} --overrides ${OVERRIDES}

interpret-debug:
	IS_DEBUG=true make interpret INTREPRET_CONFIG_PATH=config/interpret_debug.yaml 

train-then-evaluate:
	$(eval ARTIFACT_PATH := ${BASE_DIR}/log/$(shell date +'%Y%m%d%H%M%S'))
	make train TRAIN_CONFIG_PATH=${TRAIN_CONFIG_PATH} TRAIN_ARTIFACTS_DIR=${ARTIFACT_PATH} OVERRIDES='${TRAIN_OVERRIDES}'
	make evaluate EVALUATION_CONFIG_PATH=${EVALUATION_CONFIG_PATH} OVERRIDES='artifact_path=${ARTIFACT_PATH} ${EVALUATION_OVERRIDES}'
	rm -rf ${ARTIFACT_PATH}

tensorboard:
	${POETRY_RUN} tensorboard --logdir ${TENSORBORD_LOG_DIR}

# CI
lint:
	${POETRY_RUN} flake8 --show-source --statistics ./expats ./tests

test:
	${POETRY_RUN} pytest -rf --cov=./expats ./tests

typecheck:
	@echo currently not support to check types