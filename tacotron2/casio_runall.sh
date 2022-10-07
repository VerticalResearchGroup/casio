#!/bin/bash

RUN_CMD="python train.py --output_directory=outputs --log_directory=logs"

echo $RUN_CMD

for BS in 64 32 16 8 4 2 1; do
	APP=tacotron2 BS=$BS NW=10 NI=10 ../utils/run_bench.sh $RUN_CMD
	APP=tacotron2 BS=$BS NW=10 NI=10 ../utils/run_prof.sh $RUN_CMD
	APP=tacotron2 BS=$BS NW=10 NI=10 ../utils/run_nsys.sh $RUN_CMD
done


APP=tacotron2 BS=64 NW=10 NI=1 ../utils/run_ncu.sh $RUN_CMD
APP=tacotron2 BS=1 NW=10 NI=1 ../utils/run_ncu.sh $RUN_CMD
