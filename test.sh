#!/bin/bash
MODE="test"
CONTENT_FPATH="./images/content_target/rotunda.jpg"
CKPT_FPATH='./checkpoint/epoch_2_Mon_Apr_16_11:03:07_2018.model'
CKPT_FPATH2='./checkpoint/epoch_2_Mon_Apr_16_01:40:15_2018.model'

python main.py $MODE --ckpt_fpath $CKPT_FPATH --content_fpath $CONTENT_FPATH
