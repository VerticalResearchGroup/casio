#!/bin/bash

for B in 1 512 ; do
    DEV=cuda:1 python resnet50.py prof $B 30
done


