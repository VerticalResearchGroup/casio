#!/bin/bash

ODIR=$CASIO/output/$PLAT/$APP

mkdir -p $ODIR

MODE=prof $*
