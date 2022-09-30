#!/bin/bash

ODIR=$CASIO/output/$PLAT/$APP

mkdir -p $ODIR

MODE=nsys nsys profile \
    -t cuda,cudnn,cublas \
    -o $ODIR/nsys-tabnet-train-b$BS-n$NI \
    -f true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    $*

