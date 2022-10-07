#!/bin/bash

RUN_CMD="python train.py --data_dir=/work/datasets/wavenet-tiny --silence_threshold 0"

echo $RUN_CMD
#BS is ignored

APP=wavenet BS=1 NW=10 NI=10 ../utils/run_bench.sh $RUN_CMD
APP=wavenet BS=1 NW=10 NI=10 ../utils/run_prof.sh $RUN_CMD
APP=wavenet BS=1 NW=10 NI=10 ../utils/run_nsys.sh $RUN_CMD

APP=wavenet BS=1 NW=10 NI=1 ../utils/run_ncu.sh $RUN_CMD

