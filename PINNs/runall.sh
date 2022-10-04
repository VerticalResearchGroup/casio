#!/bin/bash

APPS="schrodinger ac kdv navier-stokes"

for a in $APPS ; do
    pushd $CASIO/PINNs/main/$a
    APP=pinn-$a BS=1 NW=30 NI=30 $CASIO/utils/run_bench.sh python main.py
    APP=pinn-$a BS=1 NW=30 NI=30 $CASIO/utils/run_prof.sh python main.py
    APP=pinn-$a BS=1 NW=30 NI=30 $CASIO/utils/run_nsys.sh python main.py
    APP=pinn-$a BS=1 NW=30 NI=1  $CASIO/utils/run_ncu.sh python main.py
    popd
done
