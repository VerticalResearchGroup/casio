#!/bin/bash

docker run --gpus all -it -v $PWD:/work nvcr.io/nvidia/pytorch:22.08-py3


