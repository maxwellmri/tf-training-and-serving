#!/bin/bash
set -euo pipefail
TEST_DATA=gs://maxwell-pt-test/datasets/mnist/mnist-1524490001-test.tfrecords 
TRAIN_DATA=gs://maxwell-pt-test/datasets/mnist/mnist-1524490001-train.tfrecords
JOB_NAME="Experiment7"
now=$(date +"%Y%m%d_%H:%M:%S")
LR="1e-4"
TRAIN_STEPS="20000"
EVAL_STEPS="200"
BATCH="50"
JOB_DIR=gs://ml-model-exports/mnist/deepnn/logs/$JOB_NAME
NUM_EX=55000

echo "## $JOB_NAME ($now)" >> log.md
echo "Learning Rate: $LR" >> log.md
echo "Train Steps: $TRAIN_STEPS" >> log.md
echo "Batch Size: $BATCH" >> log.md
echo "### Hypothesis
" >> log.md
echo "### Results
" >> log.md
vim + log.md

export CUDA_VISIBLE_DEVICES="0" #SET TO GPU number
python task.py \
    --train-files $TRAIN_DATA \
    --eval-files $TEST_DATA \
    --train-batch-size $BATCH \
    --eval-batch-size $BATCH \
    --max-steps $TRAIN_STEPS \
    --job-dir $JOB_DIR \
    --learning-rate $LR \
    --eval-steps $EVAL_STEPS \
    &>./runlogs/$JOB_NAME.log &

