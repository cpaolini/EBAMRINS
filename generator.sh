#!/bin/bash
DIR=/global/cfs/projectdirs/m1516/summer2023
maxJobs=100
maxGaps=9
gap=.002
n=1
while [ $n -le $maxGaps ]; do
    job=1
    gapName=${gap:1:5}
    while [ $job -le $maxJobs ]; do
	seed=$RANDOM
	if [ ! -d $DIR/testcase_${gapName}_${seed} ]; then
	    mkdir $DIR/testcase_${gapName}_${seed}
	    	cp $DIR/testcase_template/batch.perlmutter $DIR/testcase_${gapName}_${seed}/batch.perlmutter
	    	cp $DIR/testcase_template/input_test.txt $DIR/testcase_${gapName}_${seed}
	    	cp $DIR/testcase_template/.petscrc $DIR/testcase_${gapName}_${seed}
	    	sed -e "s/@RANDSEED@/${seed}/" $DIR/testcase_template/packedChannel_template.inputs > $DIR/testcase_${gapName}_${seed}/packedChannel.inputs
	    	sed -i -e "s/@GAP@/${gap}/" $DIR/testcase_${gapName}_${seed}/packedChannel.inputs
		cd $DIR/testcase_${gapName}_${seed}
		sbatch batch.perlmutter
	fi
	#echo "$gap $gapName $job"
	job=$(( $job + 1 ))
    done
    n=$(( $n + 1 ))
    gap=`echo "$gap + 0.001" | bc`
done
#ls -l $DIR
squeue -u paolini


