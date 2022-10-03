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

ODIR=$CASIO/output/$PLAT/$APP

mkdir -p $ODIR

MODE=prof $* | tee $ODIR/bench-$APP-train-b$BS-n$NI.txt
