#!/bin/bash
MODE="train"
DATASET_DIR="./dataset"
STYLE_FNAME="./style_target/thestarrynight.jpg"

python main.py $MODE --dataset $DATASET_DIR --style_fname $STYLE_FNAME
