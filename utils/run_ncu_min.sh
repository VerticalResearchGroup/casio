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

CUDNN_LIBS=/nobackup/medavies/anaconda3/envs/casio/lib/python3.9/site-packages/torch/lib/

TMPDIR=/nobackup/medavies/tmp CUDA_LAUNCH_BLOCKING=1 NW=1 NI=1 MODE=ncu LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDNN_LIBS \
    $NCU \
    $SAMP_NCU_FLAG \
    --target-processes all \
    --profile-from-start no \
    --replay-mode application \
    --metrics gpu__time_duration.sum,launch__thread_count,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --csv \
    $* | tee $ODIR/ncu-min-$SAMP-$APP-b$BS-raw.txt
