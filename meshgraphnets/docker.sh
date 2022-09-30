#!/bin/bash

docker run --gpus all -it -v /nobackup.1/medavies/casio:/work -v /nobackup.1/karu/datasets/:/datasets nvcr.io/nvidia/tensorflow:22.08-tf1-py3


