#!/bin/bash

for B in 1 512 ; do
    DEV=cuda:1 ./ncu.sh $B
done


