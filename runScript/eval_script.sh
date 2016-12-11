#########################################################################
# File Name: eval_script
# Author: qiao_yuchen
# mail: qiaoyc14@mails.tsinghua.edu.cn
# Created Time: Sun Dec 13 2016
#########################################################################
#!/bin/bash

# input & output file location
DATASET_LOCATION=../../n3lp_data/
RESULT_LOCATION=../result/evaluation

# parameters
INPUT_DIMENTION=512
HIDDEN_DIMENTION=512

DATASET_SIZE=(
	1k  
)
# for magellan
NUM_THREADS=(
	12
)
MINIBATCH_SIZE=128

# executable file
EXECUTABLE_FILE=(
	n3lp
)

# log file and static file

# const
UNDERLINE=_
datasetPath=
inputDim=$INPUT_DIMENTION
hiddenDim=$HIDDEN_DIMENTION
miniBatchSize=$MINIBATCH_SIZE

START_LOG=time_rec_start.log
END_LOG=time_rec_end.log
TIME_LOG=time_each_minibatch.log
LOG_FILE+=runtime.log

echo START!

for exec in ${EXECUTABLE_FILE[*]}
do
	for datasetSize in ${DATASET_SIZE[*]}
	do
		datasetPath=$DATASET_LOCATION
		datasetPath+=$DATASET_SIZE
		for numThreads in ${NUM_THREADS[*]}
		do
			result_path=$RESULT_LOCATION$exec$UNDERLINE$DATASET_SIZE$UNDERLINE$miniBatchSize$UNDERLINE$numThreads$UNDERLINE
			
			../$exec -d $datasetPath -i $inputDim -h $hiddenDim -m $miniBatchSize -n $numThreads -v 1 > $LOG_FILE
			cat $LOG_FILE
			result_path1=result_path
			result_path1+=1
			mv ../$START_LOG $result_path1$UNDERLINE$START_LOG
			mv ../$END_LOG $result_path1$UNDERLINE$ENDLOG
			mv ../$TIME_LOG $result_path1$UNDERLINE$TIME_LOG
			mv ../$LOG_FILE $result_path1$UNDERLINE$LOG_FILE
			
			../$exec -d $datasetPath -i $inputDim -h $hiddenDim -m $miniBatchSize -n $numThreads -v 2 > $LOG_FILE
			cat $LOG_FILE
			result_path2=result_path
			result_path1+=2	
			mv ../$START_LOG $result_path1$UNDERLINE$START_LOG
			mv ../$END_LOG $result_path1$UNDERLINE$ENDLOG
			mv ../$TIME_LOG $result_path1$UNDERLINE$TIME_LOG
			mv ../$LOG_FILE $result_path1$UNDERLINE$LOG_FILE
				
		done
	done
done

echo END!

