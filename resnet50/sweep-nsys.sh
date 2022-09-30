#!/bin/bash

for B in 1 2 4 8 16 32 64 128 256 512 ; do
    DEV=cuda:1 ./nsys.sh $B 30
done


