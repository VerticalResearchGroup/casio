#!/bin/bash

ODIR=/work/output/$PLAT/meshgraphnets

mkdir -p $ODIR

MODE=prof \
    python -m meshgraphnets.run_model \
        --mode=train \
        --model=$2 \
        --dataset_dir=/datasets/$1 \
        --checkpoint_dir=/tmp/$1-$2 --execstyle='none' \
    | tee $ODIR/prof-meshgraphnets-$1-train-b$BS.txt

