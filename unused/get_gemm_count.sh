#!/bin/sh

gzip -d < $1 | egrep -v "redzone|CUDA memset|CUDA memcpy" > /tmp/foo
#echo -n "GEMM kernels: "
x=`grep -v "Eigen" /tmp/foo | egrep -i "gemm|wgrad|nchw|nhwc|conv2d|dgrad2d|convolve" | wc -l`
echo -n $x", "
#echo -n "Total kernels: "
y=`cat /tmp/foo | wc -l`
echo -n $y", "
#echo -n "GEMM time "
a=`grep -v "Eigen" /tmp/foo | egrep -i "gemm|wgrad|nchw|nhwc|conv2d|dgrad2d|convolve" |  cut -d"," -f2 | awk '{s=s+$1}END{print s}'`
echo -n $a", "
#echo -n "Total time "
b=`cat /tmp/foo |  cut -d"," -f2 | awk '{s=s+$1}END{print s}'`
echo -n $b", " 
#echo -n "GEMM fraction by nkernels "
z=`echo $x"/"$y | bc -l`
echo -n $z", "
#echo -n "GEMM fraction by time "
z=`echo $a"/"$b | bc -l`
echo $z""


