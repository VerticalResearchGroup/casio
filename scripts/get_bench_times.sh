#!/bin/sh


for a in *; do
	if [ -d "$a" ] 
	then
		cd $a
#		grep "Throughput" bench*.txt | sed -e"s/bench.*train-//g" | sed -e"s/-n.*.txt:/ /g" | sed -e"s/^b//g" | sort -n -k1 | sed -e"s/^/$a /g"
		outdir="$1/$a"
		mkdir -p $outdir
		outfilename="$1/$a/bench-times.csv"
		if test -f "$outfilename"; then
			echo "$outfilename  exists...aborting"
#			exit 1
	 	fi 
		python3	$CASIO_ROOT/scripts/get_bench_times.py *bench* | sort -n -k1 > $outfilename
		echo "Write to "$outfilename
		cd ..
	fi
done

