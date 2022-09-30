#!/bin/bash

ODIR=/work/output/$PLAT/meshgraphnets

mkdir -p $ODIR
# --replay-mode application

T0=$(python -c 'import time ; print(time.perf_counter())')
NW=30 NI=1 MODE=ncu ncu \
    --profile-from-start no \
    --page raw \
    --set full \
    --csv \
    python -m meshgraphnets.run_model \
        --mode=train \
        --model=$2 \
        --dataset_dir=/datasets/$1 \
        --checkpoint_dir=/tmp/$1-$2 --execstyle='none' \
    | tee $ODIR/ncu-meshgraphnets-$1-train-raw.txt

python -c "import time ; print(\"Elapsed time: \", time.perf_counter() - $T0)"

MODE=ncu ncu \
    --profile-from-start no \
    --print-source=sass \
    --page source \
    --set full \
    --csv \
        python -m meshgraphnets.run_model \
        --mode=train \
        --model=$2 \
        --dataset_dir=/datasets/$1 \
        --checkpoint_dir=/tmp/$1-$2 --execstyle='none' \
    | tee $ODIR/ncu-meshgraphnets-$1-train-sass.txt
