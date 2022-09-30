#!/bin/bash

PLAT=$(python -c 'import platform ; print(platform.node())')
ODIR=$PWD/../output/$PLAT/pinn-$1

mkdir -p $ODIR

pushd main/$1
MODE=nsys nsys profile \
    -t cuda,cudnn,cublas \
    -o $ODIR/nsys-pinn-$1-train-n$NI \
    -f true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    python main.py

popd
