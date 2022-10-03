#!/bin/bash

# Env vars:
# CASIO = /path/to/casio
# APP = name of application
# MODE = {ncu, nsys, prof, bench}
# PLAT = {a100, v100, p100}
# DEV = {cuda:0, cuda:1, ...}
# BS = batch size
# NW = number of warmup steps
# NI = number of benchmark iterations

set -x
set -e

ODIR=$CASIO/output/$PLAT/$APP/

mkdir -p $ODIR

NW=1 NI=1 MODE=ncu /opt/nvidia/nsight-compute/2022.2.1/ncu \
    --profile-from-start no \
    --page raw \
    --set full \
    --csv \
    $* | tee $ODIR/ncu-$APP-train-b$BS-raw.txt

NW=1 NI=1 MODE=ncu /opt/nvidia/nsight-compute/2022.2.1/ncu \
    --profile-from-start no \
    --print-source=sass \
    --page source \
    --set full \
    --csv \
    $* | tee $ODIR/ncu-$APP-train-b$BS-sass.txt

