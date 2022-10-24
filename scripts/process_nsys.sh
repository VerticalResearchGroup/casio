#!/bin/sh

f=`mktemp -d`
oprefix="batch-"$2
cp $1 $f/$oprefix.nsys-rep
cd $f
nsys stats --report gputrace  --report gpukernsum --report cudaapisum --format csv,csv --output .,.,. $oprefix.nsys-rep
gzip $f/$oprefix"_gputrace.csv"
rm $f/$oprefix.nsys-rep
rm $f/$oprefix.sqlite

cp $f/*.csv $3
cp $f/*.csv.gz $3
echo "Removing directory $f"
rm -rf $f
