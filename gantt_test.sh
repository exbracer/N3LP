#########################################################################
# File Name: gantt_test.sh
# Author: qiao_yuchen
# mail: qiaoyc14@mails.tsinghua.edu.cn
# Created Time: Fri Jul 15 16:30:15 2016
#########################################################################
#!/bin/bash
DATASET_LOCATION=../n3lp_data/
DATASET_SIZE=$1
DATASET_LOCATION+=$DATASET_SIZE

RESULT_LOCATION=./gantt_result/
UNDERLINE=_
INPUT_DIMENTION=50
HIDDEN_DIMENTION=50
MIN_I=0
MAX_I=0
MIN_J=4
MAX_J=4

START_LOG=time_rec_start.log
END_LOG=time_rec_end.log

EXECUTABLE_FILE=(
	n3lp
	n3lp_tc
)

echo GANTT GO!!

for elem in ${EXECUTABLE_FILE[*]}
do
	echo $elem
	for ((i=$MIN_I;i<=$MAX_I;i++))
	do
		for ((j=$MIN_J;j<=$MAX_J;j++))
		do
			miniBatchSize=$[128*2**$i]
			numThreads=$[2**$j]
			./$elem -d $DATASET_LOCATION -i $INPUT_DIMENTION -h $HIDDEN_DIMENTION -m $miniBatchSize -n $numThreads 
			mv ./$START_LOG $RESULT_LOCATION$elem$UNDERLINE$DATASET_SIZE$UNDERLINE$miniBatchSize$UNDERLINE$numThreads$UNDERLINE$START_LOG
			mv ./$END_LOG $RESULT_LOCATION$elem$UNDERLINE$DATASET_SIZE$UNDERLINE$miniBatchSize$UNDERLINE$numThreads$UNDERLINE$END_LOG
		done
	done
done
