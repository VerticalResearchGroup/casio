#!/bin/bash

CLOTH_CMD="python -m meshgraphnets.run_model \
    --mode=train \
    --model=cloth \
    --dataset_dir=./meshgraphnets/datasets/flag_simple \
    --checkpoint_dir=/tmp/flag_simple-cloth --execstyle='none'"

echo $CLOTH_CMD

CFD_CMD="python -m meshgraphnets.run_model \
    --mode=train \
    --model=cfd \
    --dataset_dir=./meshgraphnets/datasets/cylinder_flow \
    --checkpoint_dir=/tmp/cylinder_flow-cfd --execstyle='none'"

echo $CFD_CMD

./meshgraphnets/ckpt.sh flag_simple cloth
APP=meshgraphnets-cloth BS=1 NW=30 NI=30 ./../utils/run_bench.sh $CLOTH_CMD
APP=meshgraphnets-cloth BS=1 NW=30 NI=30 ./../utils/run_prof.sh $CLOTH_CMD
APP=meshgraphnets-cloth BS=1 NW=30 NI=30 ./../utils/run_nsys.sh $CLOTH_CMD

./meshgraphnets/ckpt.sh cylinder_flow cfd
APP=meshgraphnets-cfd BS=1 NW=30 NI=30 ./../utils/run_bench.sh $CFD_CMD
APP=meshgraphnets-cfd BS=1 NW=30 NI=30 ./../utils/run_prof.sh $CFD_CMD
APP=meshgraphnets-cfd BS=1 NW=30 NI=30 ./../utils/run_nsys.sh $CFD_CMD

APP=meshgraphnets-cloth BS=1 NW=30 NI=1 ./../utils/run_nsu.sh $CLOTH_CMD
APP=meshgraphnets-cfd BS=1 NW=30 NI=1 ./../utils/run_nsu.sh $CFD_CMD
