#!/bin/bash
set -x
set -e

CMD="python test_experiment_covertype.py"

echo "Patching python env..."
./patch.sh

APP=tabnet BS=4 NI=30 ./../utils/run_bench.sh $CMD
APP=tabnet BS=8 NI=30 ./../utils/run_bench.sh $CMD
APP=tabnet BS=16 NI=30 ./../utils/run_bench.sh $CMD
APP=tabnet BS=32 NI=30 ./../utils/run_bench.sh $CMD
APP=tabnet BS=64 NI=30 ./../utils/run_bench.sh $CMD
APP=tabnet BS=128 NI=30 ./../utils/run_bench.sh $CMD

APP=tabnet BS=4 NI=30 ./../utils/run_prof.sh $CMD
APP=tabnet BS=128 NI=30 ./../utils/run_prof.sh $CMD

APP=tabnet BS=4 ./../utils/run_nsys.sh $CMD
APP=tabnet BS=8 ./../utils/run_nsys.sh $CMD
APP=tabnet BS=16 ./../utils/run_nsys.sh $CMD
APP=tabnet BS=32 ./../utils/run_nsys.sh $CMD
APP=tabnet BS=64 ./../utils/run_nsys.sh $CMD
APP=tabnet BS=128 ./../utils/run_nsys.sh $CMD

APP=tabnet BS=4 ./../utils/run_ncu.sh $CMD
APP=tabnet BS=128 ./../utils/run_ncu.sh $CMD
