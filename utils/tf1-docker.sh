#!/bin/bash

docker run --gpus all -it -v $CASIO:/work nvcr.io/nvidia/tensorflow:22.08-tf1-py3


