#!/bin/bash

mkdir -p ../output

/opt/nvidia/nsight-systems/2022.1.3/bin/nsys profile \
    -t cuda,cudnn,cublas \
    -o ../output/nsys-resnet50-train-b$1-n$2 \
    -f true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    python resnet50.py nsys $1 $2

