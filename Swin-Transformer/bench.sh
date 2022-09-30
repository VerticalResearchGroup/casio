#!/bin/bash


ODIR=$CASIO/output/$PLAT/swin-$1

mkdir -p $ODIR

MODE=bench python main.py --cfg configs/swinv2/$1.yaml | tee $ODIR/bench-swin-$1-train-b$BS.txt

