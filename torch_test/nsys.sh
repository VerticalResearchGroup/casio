#!/bin/bash

nsys profile -t cuda,cudnn,cublas -o dump --capture-range=cudaProfilerApi  --stop-on-range-end=true python test.py

# nsys profile -t cuda,cudnn,cublas -o dump -f true  python tftest.py
