#!/bin/sh


for a in *; do
	if [ -d "$a" ] 
	then
		cd $a
		echo "Entering $a"
		sassdir="$1/$a/sass"
		outdir="$1/$a"
		for y in *ncu-10th-*sass.txt; do
			batch_size=`python3 $CASIO_ROOT/scripts/get_batch_size_from_name.py $y`
			echo $y" "$batch_size
			nkernels=`ls $sassdir"/" | wc -l` 
			pwd
			echo python3 $CASIO_ROOT/scripts/kernel_sass_similarity.py $sassdir"/batch-"$batch_size"_" $nkernels $outdir"/batch-"$batch_size"_sass_kernel_similarity.txt"
		done
		cd ..
	fi
done

