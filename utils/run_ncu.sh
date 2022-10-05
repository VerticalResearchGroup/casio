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

RUN_NCU=${RUN_NCU:-yes}
[ "$RUN_NCU" = "no" ] && exit 0

ODIR=$CASIO/output/$PLAT/$APP/

mkdir -p $ODIR

SAMP=${SAMP:-all}

case $SAMP in
    all)
        SAMP_NCU_FLAG=""
        ;;
    10th)
        SAMP_NCU_FLAG='--kernel-id :::0|.*0'
        ;;
    20th)
        SAMP_NCU_FLAG='--kernel-id :::0|.*(2|4|6|8|0)0'
        ;;
    50th)
        SAMP_NCU_FLAG='--kernel-id :::0|.*(0|5)0'
        ;;
    100th)
        SAMP_NCU_FLAG='--kernel-id :::0|.*00'
        ;;
    *)
        echo "Unknown sampling mode: $SAMP"
        exit 1
        ;;
esac

NCU=/opt/nvidia/nsight-compute/2022.2.1/ncu
[ -x "$NCU" ] || NCU=/usr/local/cuda/bin/ncu
[ -x "$NCU" ] || NCU=ncu
echo "Using ncu: $NCU"

NW=1 NI=1 MODE=ncu $NCU \
    $SAMP_NCU_FLAG \
    --target-processes all \
    --profile-from-start no \
    --page raw \
    --set full \
    --csv \
    $* | tee $ODIR/ncu-$SAMP-$APP-train-b$BS-raw.txt

NW=1 NI=1 MODE=ncu $NCU \
    $SAMP_NCU_FLAG \
    --target-processes all \
    --profile-from-start no \
    --print-source=sass \
    --page source \
    --set full \
    --csv \
    $* | tee $ODIR/ncu-$SAMP-$APP-train-b$BS-sass.txt

