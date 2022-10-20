#!/bin/sh


for a in *; do
	if [ -d "$a" ] 
	then
		cd $a
		echo "Entering $a"
		outdir="$1/$a/sass"
		mkdir -p $outdir
		for y in *ncu-10th-*sass.txt; do
			batch_size=`python3 $CASIO_ROOT/scripts/get_batch_size_from_name.py $y`
			echo $y" "$batch_size
			python3 $CASIO_ROOT/scripts/process_sass.py $y $outdir"/batch-"$batch_size"_"
		done
		cd ..
	fi
done

