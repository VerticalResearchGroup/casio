#!/bin/bash

echo "=============================================="
echo "REMEMBER: RUN THIS INSIDE THE DOCKER CONTAINER"
echo "FOR TENSORFLOW v1 APPLICATIONS!"
echo ""
echo "MIKE WILL NOT ANSWER THIS QUESTION!"
echo "=============================================="

export CASIO=$PWD

echo -n "What platform is this? (cpu, p100, v100, a100): "
read PLAT
export PLAT

echo -n "What gpu should we use? (cuda:0, cuda:1, ...): "
read DEV
export DEV

echo -n "What level of sampling do you want? (100th, 10th, all): "
read SAMP
export SAMP

echo "Path to CASIO: $CASIO"
echo "Platform: $PLAT"
echo "Device: $DEV"
echo "Sampling: $SAMP"

