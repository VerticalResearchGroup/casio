#!/bin/bash

PLAT=$(python -c 'import platform ; print(platform.node())')
ODIR=$PWD/../output/$PLAT/pinn-$1

mkdir -p $ODIR

pushd main/$1
MODE=ncu ncu \
    --profile-from-start no \
    --page raw \
    --set full \
    --csv \
    python main.py | tee $ODIR/ncu-pinn-$1-train-raw.txt

MODE=ncu ncu \
    --profile-from-start no \
    --print-source=sass \
    --page source \
    --set full \
    --csv \
    python main.py | tee $ODIR/ncu-pinn-$1-train-sass.txt

popd
