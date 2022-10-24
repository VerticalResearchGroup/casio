#!/bin/sh


for a in *; do
	if [ -d "$a" ] 
	then
		cd $a
		echo "Entering $a"
		outdir="$1/$a"
		mkdir -p $outdir
		for y in *ncu-10th-*raw.txt; do
			batch_size=`python3 $CASIO_ROOT/scripts/get_batch_size_from_name.py $y`
			echo $y" "$batch_size
			python3 $CASIO_ROOT/scripts/histo_from_ncu.py $y $y $outdir"/batch-"$batch_size"_"
		done
		cd ..
	fi
done

