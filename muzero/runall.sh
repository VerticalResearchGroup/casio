#!/bin/bash
set -x
set -e

CMD="python minimuzero.py atari"

for i in 1 4 16 64 256 1024 ; do
    APP=muzero BS=$i NI=30 ./../utils/run_bench.sh $CMD
    APP=muzero BS=$i NI=30 ./../utils/run_prof.sh $CMD
    APP=muzero BS=$i NI=30 ./../utils/run_nsys.sh $CMD
done

for i in 1 1024 ; do
    APP=muzero BS=$i NI=30 ./../utils/run_ncu.sh $CMD
done
