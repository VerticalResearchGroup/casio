#!/bin/bash


PLAT=$(python -c 'import platform ; print(platform.node())')
ODIR=$PWD/../output/$PLAT/pinn-$1

mkdir -p $ODIR

pushd main/$1
MODE=bench python main.py | tee $ODIR/bench-pinn-$1-train-b$BS.txt
popd
