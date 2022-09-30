#!/bin/bash

ODIR=$CASIO/output/$PLAT/swin-$1
mkdir -p $ODIR

MODE=nsys /opt/nvidia/nsight-systems/2022.1.3/bin/nsys profile \
    -t cuda,cudnn,cublas \
    -o $ODIR/nsys-swin-$1-train-b$BS-n$NI \
    -f true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    python main.py --cfg configs/swinv2/$1.yaml

