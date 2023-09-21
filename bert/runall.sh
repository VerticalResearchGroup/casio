#!/bin/bash
APPNAME=bert
INFER_CMD="python -m bert.infer"
TRAIN_CMD="python -m bert.train"

INFER_DT=FP16
TRAIN_DT=FP16

I_SMALL_BS=1
I_LARGE_BS=32

T_SMALL_BS=1
T_LARGE_BS=32

APP=$APPNAME-infer DT=$INFER_DT BS=$I_SMALL_BS NW=30 NI=30 $CASIO/utils/run_bench.sh $INFER_CMD
APP=$APPNAME-infer DT=$INFER_DT BS=$I_SMALL_BS NW=30 NI=30 $CASIO/utils/run_prof.sh $INFER_CMD
APP=$APPNAME-infer DT=$INFER_DT BS=$I_SMALL_BS NW=30 NI=30 $CASIO/utils/run_nsys.sh $INFER_CMD

APP=$APPNAME-infer DT=$INFER_DT BS=$I_LARGE_BS NW=30 NI=30 $CASIO/utils/run_bench.sh $INFER_CMD
APP=$APPNAME-infer DT=$INFER_DT BS=$I_LARGE_BS NW=30 NI=30 $CASIO/utils/run_prof.sh $INFER_CMD
APP=$APPNAME-infer DT=$INFER_DT BS=$I_LARGE_BS NW=30 NI=30 $CASIO/utils/run_nsys.sh $INFER_CMD


APP=$APPNAME-train DT=$TRAIN_DT BS=$T_SMALL_BS NW=30 NI=30 $CASIO/utils/run_bench.sh $TRAIN_CMD
APP=$APPNAME-train DT=$TRAIN_DT BS=$T_SMALL_BS NW=30 NI=30 $CASIO/utils/run_prof.sh $TRAIN_CMD
APP=$APPNAME-train DT=$TRAIN_DT BS=$T_SMALL_BS NW=30 NI=30 $CASIO/utils/run_nsys.sh $TRAIN_CMD

APP=$APPNAME-train DT=$TRAIN_DT BS=$T_LARGE_BS NW=30 NI=30 $CASIO/utils/run_bench.sh $TRAIN_CMD
APP=$APPNAME-train DT=$TRAIN_DT BS=$T_LARGE_BS NW=30 NI=30 $CASIO/utils/run_prof.sh $TRAIN_CMD
APP=$APPNAME-train DT=$TRAIN_DT BS=$T_LARGE_BS NW=30 NI=30 $CASIO/utils/run_nsys.sh $TRAIN_CMD

APP=$APPNAME-infer DT=$INFER_DT BS=$I_SMALL_BS NW=30 NI=1  $CASIO/utils/run_ncu_min.sh $INFER_CMD
APP=$APPNAME-infer DT=$INFER_DT BS=$I_LARGE_BS NW=30 NI=1  $CASIO/utils/run_ncu_min.sh $INFER_CMD
APP=$APPNAME-train DT=$TRAIN_DT BS=$T_SMALL_BS NW=30 NI=1  $CASIO/utils/run_ncu_min.sh $TRAIN_CMD
APP=$APPNAME-train DT=$TRAIN_DT BS=$T_LARGE_BS NW=30 NI=1  $CASIO/utils/run_ncu_min.sh $TRAIN_CMD
