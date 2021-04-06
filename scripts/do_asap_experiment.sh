#/bin/bash

TRAIN_CONFIG=$1
EVALUATION_CONFIG=$2

echo "\n----- prompt 1 -----\n"
make train-then-evaluate \
    TRAIN_CONFIG_PATH=${TRAIN_CONFIG} \
    TRAIN_OVERRIDES='dataset.params.prompt_id=1' \
    EVALUATION_CONFIG_PATH=${EVALUATION_CONFIG} \
    EVALUATION_OVERRIDES='dataset.params.prompt_id=1 output_convert.regression_to_classification.params.x_min=2 output_convert.regression_to_classification.params.x_max=12'
echo "\n----- prompt 2 -----\n"
make train-then-evaluate \
    TRAIN_CONFIG_PATH=${TRAIN_CONFIG} \
    TRAIN_OVERRIDES='dataset.params.prompt_id=2' \
    EVALUATION_CONFIG_PATH=${EVALUATION_CONFIG} \
    EVALUATION_OVERRIDES='dataset.params.prompt_id=2 output_convert.regression_to_classification.params.x_min=1 output_convert.regression_to_classification.params.x_max=6'
echo "\n----- prompt 3 -----\n"
make train-then-evaluate \
    TRAIN_CONFIG_PATH=${TRAIN_CONFIG} \
    TRAIN_OVERRIDES='dataset.params.prompt_id=3' \
    EVALUATION_CONFIG_PATH=${EVALUATION_CONFIG} \
    EVALUATION_OVERRIDES='dataset.params.prompt_id=3 output_convert.regression_to_classification.params.x_min=0 output_convert.regression_to_classification.params.x_max=3'
echo "\n----- prompt 4 -----\n"
make train-then-evaluate \
    TRAIN_CONFIG_PATH=${TRAIN_CONFIG} \
    TRAIN_OVERRIDES='dataset.params.prompt_id=4' \
    EVALUATION_CONFIG_PATH=${EVALUATION_CONFIG} \
    EVALUATION_OVERRIDES='dataset.params.prompt_id=4 output_convert.regression_to_classification.params.x_min=0 output_convert.regression_to_classification.params.x_max=3'
echo "\n----- prompt 5 -----\n"
make train-then-evaluate \
    TRAIN_CONFIG_PATH=${TRAIN_CONFIG} \
    TRAIN_OVERRIDES='dataset.params.prompt_id=5' \
    EVALUATION_CONFIG_PATH=${EVALUATION_CONFIG} \
    EVALUATION_OVERRIDES='dataset.params.prompt_id=5 output_convert.regression_to_classification.params.x_min=0 output_convert.regression_to_classification.params.x_max=4'
echo "\n----- prompt 6 -----\n"
make train-then-evaluate \
    TRAIN_CONFIG_PATH=${TRAIN_CONFIG} \
    TRAIN_OVERRIDES='dataset.params.prompt_id=6' \
    EVALUATION_CONFIG_PATH=${EVALUATION_CONFIG} \
    EVALUATION_OVERRIDES='dataset.params.prompt_id=6 output_convert.regression_to_classification.params.x_min=0 output_convert.regression_to_classification.params.x_max=4'
echo "\n----- prompt 7 -----\n"
make train-then-evaluate \
    TRAIN_CONFIG_PATH=${TRAIN_CONFIG} \
    TRAIN_OVERRIDES='dataset.params.prompt_id=7' \
    EVALUATION_CONFIG_PATH=${EVALUATION_CONFIG} \
    EVALUATION_OVERRIDES='dataset.params.prompt_id=7 output_convert.regression_to_classification.params.x_min=2 output_convert.regression_to_classification.params.x_max=24'
echo "\n----- prompt 8 -----\n"
make train-then-evaluate \
    TRAIN_CONFIG_PATH=${TRAIN_CONFIG} \
    TRAIN_OVERRIDES='dataset.params.prompt_id=8' \
    EVALUATION_CONFIG_PATH=${EVALUATION_CONFIG} \
    EVALUATION_OVERRIDES='dataset.params.prompt_id=8 output_convert.regression_to_classification.params.x_min=10 output_convert.regression_to_classification.params.x_max=60'
