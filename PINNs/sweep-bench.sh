#!/bin/bash

set -e

BS=1 NW=100 NI=1 ./bench.sh schrodinger
BS=1 NW=100 NI=1 ./bench.sh navier-stokes
BS=1 NW=100 NI=1 ./bench.sh ac
BS=1 NW=100 NI=1 ./bench.sh kdv
