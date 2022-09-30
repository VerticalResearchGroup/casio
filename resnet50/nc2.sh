#!/bin/bash

mkdir -p ../output

ncu \
    --profile-from-start no \
    --print-source=sass \
    --page=source \
    --set full \
    --csv \
    python resnet50.py $1 1 | tee ../output/ncu-resnet50-train-b$1-sass.txt

