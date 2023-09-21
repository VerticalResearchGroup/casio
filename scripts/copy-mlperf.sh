#!/bin/bash

[ "$PLAT" = "" ] && echo "ERROR: SET \$PLAT!!!" &&  exit 0

set -x
set -e

rsync -a output/$PLAT/resnet50-infer/ casio-results/$PLAT/resnet50-infer
rsync -a output/$PLAT/resnet50-train/ casio-results/$PLAT/resnet50-train
rsync -a output/$PLAT/ssdrn34-infer/ casio-results/$PLAT/ssdrn34-infer
rsync -a output/$PLAT/ssdrn34-train/ casio-results/$PLAT/ssdrn34-train
rsync -a output/$PLAT/bert-infer/ casio-results/$PLAT/bert-infer
rsync -a output/$PLAT/bert-train/ casio-results/$PLAT/bert-train
rsync -a output/$PLAT/bert-infer/ casio-results/$PLAT/bert-infer
rsync -a output/$PLAT/bert-train/ casio-results/$PLAT/bert-train
rsync -a output/$PLAT/dlrm-infer/ casio-results/$PLAT/dlrm-infer
rsync -a output/$PLAT/dlrm-train/ casio-results/$PLAT/dlrm-train
rsync -a output/$PLAT/unet-infer/ casio-results/$PLAT/unet-infer
rsync -a output/$PLAT/unet-train/ casio-results/$PLAT/unet-train
