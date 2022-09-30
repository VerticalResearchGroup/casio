#!/bin/bash

# Env vars:
# MODE = {ncu, nsys, prof, bench}
# PLAT = {a100, v100, p100}
# DEV = {cuda:0, cuda:1, ...}
# BS = batch size
# NW = number of warmup steps
# NI = number of benchmark iterations

ODIR=/nobackup/medavies/casio/output/$PLAT/qdtrack/

mkdir -p $ODIR

MODE=ncu /opt/nvidia/nsight-compute/2022.2.1/ncu \
    --kernel-id :::"1|.*00" \
    --profile-from-start no \
    --page raw \
    --set full \
    --csv \
    python tools/train.py configs/bdd100k/qdtrack-frcnn_r50_fpn_12e_bdd100k.py --no-validate | tee $ODIR/ncu-qdtrack-train-b$BS-raw.txt

MODE=ncu /opt/nvidia/nsight-compute/2022.2.1/ncu \
    --kernel-id :::"1|.*00" \
    --profile-from-start no \
    --print-source=sass \
    --page source \
    --set full \
    --csv \
    python tools/train.py configs/bdd100k/qdtrack-frcnn_r50_fpn_12e_bdd100k.py --no-validate | tee $ODIR/ncu-qdtrack-train-b$BS-sass.txt

