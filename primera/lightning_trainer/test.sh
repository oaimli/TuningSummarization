#!/bin/bash

DATA_NAME="multinews"
PLM_MODEL_PATH="allenai/PRIMERA-multinews"
LENGTH_INPUT=4096
LENGTH_TGT=1024


python primera.py  \
                --batch_size 2 \
                --devices 1  \
                --mode test \
                --model_path result/PRIMERA_${DATA_NAME}_${LENGTH_INPUT}_${LENGTH_TGT}/ \
                --dataset_name ${DATA_NAME} \
                --pretrained_model ${PLM_MODEL_PATH} \
                --num_workers 2 \
                --beam_size 5 \
                --num_test_data 512 \
                --resume_ckpt step=24988-vloss=3.14-avgf=0.3267.ckpt