#!/bin/bash

DATASET_NAME="multinews"

PLM_MODEL_PATH="allenai/PRIMERA"
LENGTH_INPUT=4096
LENGTH_TGT=1024

python primera.py  \
                --batch_size 1 \
                --devices 2  \
                --accelerator gpu \
                --speed_strategy ddp_find_unused_parameters_false \
                --mode train \
                --model_path result/PRIMERA_${DATASET_NAME}_${LENGTH_INPUT}_${LENGTH_TGT}/ \
                --dataset_name ${DATASET_NAME} \
                --pretrained_model ${PLM_MODEL_PATH} \
                --num_workers 8 \
                --beam_size 5 \
                --test_imediate \
                --adafactor \
                --total_steps 10000000 \
                --label_smoothing 0.1 \
                --accum_data_per_step 8 \
                --val_check_interval 500 \
                --num_train_data -1 \
                --num_val_data 256 \
                --num_test_data 256