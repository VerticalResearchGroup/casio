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
        SAMP_NCU_FLAG='--kernels :::0|.*0'
        ;;
    20th)
        SAMP_NCU_FLAG='--kernels :::0|.*(2|4|6|8|0)0'
        ;;
    50th)
        SAMP_NCU_FLAG='--kernels :::0|.*(0|5)0'
        ;;
    100th)
        SAMP_NCU_FLAG='--kernels :::0|.*00'
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

# --profile-from-start off  --profile-all-processes --events all 
# --analysis-metrics -a instruction_execution -o /tmp/foobar 
NVPROF=/usr/local/cuda/bin/nvprof
echo "Using nvprof: $NVPROF"
NI=1 MODE=ncu $NVPROF \
    $SAMP_NCU_FLAG \
    --profile-child-processes  \
    --profile-from-start off \
    --events all \
    --csv \
    $* 2>&1| tee $ODIR/nvprof-$SAMP-$APP-train-b$BS-raw.txt

NI=1 MODE=ncu $NVPROF \
    $SAMP_NCU_FLAG \
    --profile-child-processes \
    --profile-from-start off \
    --analysis-metrics \
    -a instruction_execution \
    -o  $ODIR/nvprof-$SAMP-$APP-train-b$BS--sass-nvvp%p \
    $* 2>&1| tee $ODIR/nvprof-$SAMP-$APP-train-b$BS-sass.txt

