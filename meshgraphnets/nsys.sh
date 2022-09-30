#!/bin/bash

ODIR=/work/output/$PLAT/meshgraphnets

mkdir -p $ODIR


MODE=nsys nsys profile \
    -t cuda,cudnn,cublas \
    -o $ODIR/nsys-meshgraphnets-$1-train-n$NI \
    -f true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    python -m meshgraphnets.run_model \
        --mode=train \
        --model=$2 \
        --dataset_dir=/datasets/$1 \
        --checkpoint_dir=/tmp/$1-$2 --execstyle='none'


