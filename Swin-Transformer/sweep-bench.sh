#!/bin/bash

for B in 1 2 4 8 16 32 64; do
    BS=$B NI=30 ./bench.sh swinv2_base_patch4_window16_256
done

for B in 1 2 4 8 16 32; do
    BS=$B NI=30 ./bench.sh swinv2_base_patch4_window12_192_22k
done

for B in 1 2 4 8 16; do
    BS=$B NI=30 ./bench.sh swinv2_large_patch4_window12_192_22k
done

for B in 1 2; do
    BS=$B NI=30 ./bench.sh swinv2_large_patch4_window12to24_192to384_22kto1k_ft
done

