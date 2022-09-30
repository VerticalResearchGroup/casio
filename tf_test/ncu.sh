#!/bin/bash

MODE=ncu NW=30 NI=1 ncu \
    --page raw \
    --set full \
    --csv \
    python tftest.py

