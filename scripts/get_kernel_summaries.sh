#!/bin/sh

f1=`mktemp`
f2=`mktemp`
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
			python3 $CASIO_ROOT/scripts/kernel_similarity.py $y > $outdir"/batch-"$batch_size"_ncu_kernel_similarity.txt"
			grep -A1 "Process ID" $y > $f1
			grep "127.0.0.1" $y >> $f2
		done
		cd ..
	fi
done

echo "Merging all"
f3=`mktemp`
cat $f1 $f2 > $f3
python3 $CASIO_ROOT/scripts/kernel_similarity.py $f3 > $1"/all_ncu_kernel_similarity.txt"
cp $f3 /tmp/newfoo
rm $f1 $f2 $f3
