#!/bin/bash

# APP=gpt3 BS=1 NW=30 NI=30 $CASIO/utils/run_bench.sh python tests/casio_bench.py
# APP=gpt3 BS=1 NW=30 NI=30 $CASIO/utils/run_prof.sh python tests/casio_bench.py
APP=gpt3 BS=1 NW=30 NI=30 $CASIO/utils/run_nsys.sh python tests/casio_bench.py
APP=gpt3 BS=1 NW=30 NI=1  $CASIO/utils/run_ncu.sh python tests/casio_bench.py
