#!/bin/bash

# ./meshgraphnets/ckpt.sh flag_simple cloth
# BS=1 NW=30 NI=30 ./meshgraphnets/bench.sh flag_simple cloth
# BS=1 NW=30 NI=30 ./meshgraphnets/prof.sh  flag_simple cloth
# BS=1 NW=30 NI=30 ./meshgraphnets/nsys.sh  flag_simple cloth

# ./meshgraphnets/ckpt.sh cylinder_flow cfd
# BS=1 NW=30 NI=30 ./meshgraphnets/bench.sh cylinder_flow cfd
# BS=1 NW=30 NI=30 ./meshgraphnets/prof.sh  cylinder_flow cfd
# BS=1 NW=30 NI=30 ./meshgraphnets/nsys.sh  cylinder_flow cfd
BS=1 NW=1 NI=1  ./meshgraphnets/ncu.sh   cylinder_flow cfd
BS=1 NW=1 NI=1  ./meshgraphnets/ncu.sh   flag_simple cloth
