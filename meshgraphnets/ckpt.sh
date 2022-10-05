#!/bin/bash

python -m meshgraphnets.run_model \
    --mode=train --model=$2 \
    --dataset_dir=./meshgraphnets/datasets/$1 \
    --checkpoint_dir=/tmp/$1-$2 --execstyle='createchkpoint' \
    --num_training_steps=10000


