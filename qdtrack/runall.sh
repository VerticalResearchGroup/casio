#!/bin/bash
set -x
set -e

CMD="python tools/train.py configs/bdd100k/qdtrack-frcnn_r50_fpn_12e_bdd100k.py"

APP=qdtrack BS=1  NI=30 ./../utils/run_bench.sh $CMD
APP=qdtrack BS=2  NI=30 ./../utils/run_bench.sh $CMD

APP=qdtrack BS=1  NI=30 ./../utils/run_prof.sh $CMD
APP=qdtrack BS=2  NI=30 ./../utils/run_prof.sh $CMD

APP=qdtrack BS=1  NI=30 ./../utils/run_nsys.sh $CMD
APP=qdtrack BS=2  NI=30 ./../utils/run_nsys.sh $CMD

APP=qdtrack BS=1  NI=1 ./../utils/run_ncu.sh $CMD
APP=qdtrack BS=2  NI=1 ./../utils/run_ncu.sh $CMD
