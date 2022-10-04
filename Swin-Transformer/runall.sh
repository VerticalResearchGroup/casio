#!/bin/bash

SW2B16_CMD="python main.py --cfg configs/swinv2/swinv2_base_patch4_window16_256.yaml"
SW2B12FT_CMD="python main.py --cfg configs/swinv2/swinv2_base_patch4_window12_192_22k.yaml"
SW2L12_CMD="python main.py --cfg configs/swinv2/swinv2_large_patch4_window12_192_22k.yaml"
SW2L12FT_CMD="python main.py --cfg configs/swinv2/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.yaml"


for B in 1 2 4 8 16 32; do
    APP=swin-swinv2_base_patch4_window16_256 BS=$B NI=30 $CASIO/utils/run_bench.sh $SW2B16_CMD
    APP=swin-swinv2_base_patch4_window16_256 BS=$B NI=30 $CASIO/utils/run_prof.sh $SW2B16_CMD
    APP=swin-swinv2_base_patch4_window16_256 BS=$B NI=30 $CASIO/utils/run_nsys.sh $SW2B16_CMD
done

for B in 1 2 4 8 16 32; do
    APP=swin-swinv2_base_patch4_window12_192_22k BS=$B NI=30 $CASIO/utils/run_bench.sh $SW2B12FT_CMD
    APP=swin-swinv2_base_patch4_window12_192_22k BS=$B NI=30 $CASIO/utils/run_prof.sh $SW2B12FT_CMD
    APP=swin-swinv2_base_patch4_window12_192_22k BS=$B NI=30 $CASIO/utils/run_nsys.sh $SW2B12FT_CMD
done

for B in 1 2 4 8 16; do
    APP=swin-swinv2_large_patch4_window12_192_22k BS=$B NI=30 $CASIO/utils/run_bench.sh $SW2L12_CMD
    APP=swin-swinv2_large_patch4_window12_192_22k BS=$B NI=30 $CASIO/utils/run_prof.sh $SW2L12_CMD
    APP=swin-swinv2_large_patch4_window12_192_22k BS=$B NI=30 $CASIO/utils/run_nsys.sh $SW2L12_CMD
done

for B in 1 2 4 8 16; do
    APP=swin-swinv2_large_patch4_window12to24_192to384_22kto1k_ft BS=$B NI=30 $CASIO/utils/run_bench.sh $SW2L12FT_CMD
    APP=swin-swinv2_large_patch4_window12to24_192to384_22kto1k_ft BS=$B NI=30 $CASIO/utils/run_prof.sh $SW2L12FT_CMD
    APP=swin-swinv2_large_patch4_window12to24_192to384_22kto1k_ft BS=$B NI=30 $CASIO/utils/run_nsys.sh $SW2L12FT_CMD
done

########## NCU


for B in 1 32; do
    APP=swin-swinv2_base_patch4_window16_256 BS=$B NI=1 $CASIO/utils/run_ncu.sh $SW2B16_CMD
done

for B in 1 32; do
    APP=swin-swinv2_base_patch4_window12_192_22k BS=$B NI=1 $CASIO/utils/run_ncu.sh $SW2B12FT_CMD
done

for B in 1 16; do
    APP=swin-swinv2_large_patch4_window12_192_22k BS=$B NI=1 $CASIO/utils/run_ncu.sh $SW2L12_CMD
done

for B in 1 16; do
    APP=swin-swinv2_large_patch4_window12to24_192to384_22kto1k_ft BS=$B NI=1 $CASIO/utils/run_ncu.sh $SW2L12FT_CMD
done
