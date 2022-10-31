#!/bin/sh

f1=`mktemp`
f2=`mktemp`
f3=`mktemp`
for a in `cat a100-large-batch-list`; do echo -n $a","; $CASIO_ROOT/scripts/get_gemm_count.sh $a"_gputrace.csv.gz"; done > $f1
for a in `cat v100-large-batch-list`; do echo -n $a","; $CASIO_ROOT/scripts/get_gemm_count.sh $a"_gputrace.csv.gz"; done > $f2
for a in `cat p100-large-batch-list`; do echo -n $a","; $CASIO_ROOT/scripts/get_gemm_count.sh $a"_gputrace.csv.gz"; done > $f3
echo "app,gemm-count,total-count,gemm-time,total-time,count-%,time-%,app,gemm-count,total-count,gemm-time,total-time,count-%,time-%,app,gemm-count,total-count,gemm-time,total-time,count-%,time-%"
paste -d"," $f1 $f2 $f3


for a in `cat a100-small-batch-list`; do echo -n $a","; $CASIO_ROOT/scripts/get_gemm_count.sh $a"_gputrace.csv.gz"; done > $f1
for a in `cat v100-small-batch-list`; do echo -n $a","; $CASIO_ROOT/scripts/get_gemm_count.sh $a"_gputrace.csv.gz"; done > $f2
for a in `cat p100-small-batch-list`; do echo -n $a","; $CASIO_ROOT/scripts/get_gemm_count.sh $a"_gputrace.csv.gz"; done > $f3
echo "app,gemm-count,total-count,gemm-time,total-time,count-%,time-%,app,gemm-count,total-count,gemm-time,total-time,count-%,time-%,app,gemm-count,total
-count,gemm-time,total-time,count-%,time-%"
paste -d"," $f1 $f2 $f3

rm $f1 $f2 $f3


