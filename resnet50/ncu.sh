#!/bin/bash

mkdir -p ../output

/opt/nvidia/nsight-compute/2022.2.1/ncu \
    --profile-from-start no \
    --page raw \
    --set full \
    --csv \
    python resnet50.py ncu $1 1 | tee ../output/ncu-resnet50-train-b$1-raw.txt

/opt/nvidia/nsight-compute/2022.2.1/ncu \
    --profile-from-start no \
    --print-source=sass \
    --page source \
    --set full \
    --csv \
    python resnet50.py ncu $1 1 | tee ../output/ncu-resnet50-train-b$1-sass.txt

