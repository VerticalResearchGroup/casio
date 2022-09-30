#!/bin/bash

BS=1 NW=100 NI=100 ./nsys.sh schrodinger
BS=1 NW=100 NI=100 ./nsys.sh navier-stokes
BS=1 NW=100 NI=100 ./nsys.sh ac
BS=1 NW=100 NI=100 ./nsys.sh kdv
