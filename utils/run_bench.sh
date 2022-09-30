#!/bin/bash

ODIR=$CASIO/output/$PLAT/$APP

mkdir -p $ODIR

MODE=prof $* | tee $ODIR/bench-$APP-train-b$BS-n$NI.txt
