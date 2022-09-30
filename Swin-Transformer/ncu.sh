#!/bin/bash

ODIR=$CASIO/output/$PLAT/swin-$1
mkdir -p $ODIR

MODE=ncu /opt/nvidia/nsight-compute/2022.2.1/ncu \
    --profile-from-start no \
    --page raw \
    --set full \
    --csv \
    python main.py --cfg configs/swinv2/$1.yaml | tee $ODIR/ncu-swin-$1-train-b$BS-raw.txt

MODE=ncu /opt/nvidia/nsight-compute/2022.2.1/ncu \
    --profile-from-start no \
    --print-source=sass \
    --page source \
    --set full \
    --csv \
    python main.py --cfg configs/swinv2/$1.yaml | tee $ODIR/ncu-swin-$1-train-b$BS-sass.txt
