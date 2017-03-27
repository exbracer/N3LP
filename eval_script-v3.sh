#########################################################################
# File Name: eval_script
# Author: qiao_yuchen
# mail: qiaoyc14@mails.tsinghua.edu.cn
# Created Time: Sun Dec 13 2016
#########################################################################
#!/bin/bash

# input & output file location
DATASET_LOCATION=../n3lp_data/
RESULT_LOCATION=./result/evaluation/andromeda-2/

# parameters
INPUT_DIMENTION=512
HIDDEN_DIMENTION=512

DATASET_SIZE=( 
	1k
	4k
	10k
)
# for magellan
NUM_THREADS=(
	1
	4
	8
	12
	16
	48
	64
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
LOG_FILE=runtime.log

echo START!

for exec in ${EXECUTABLE_FILE[*]}
do
	for datasetSize in ${DATASET_SIZE[*]}
	do
		datasetPath=$DATASET_LOCATION
		datasetPath+=$datasetSize
		for numThreads in ${NUM_THREADS[*]}
		do
			result_path=$RESULT_LOCATION$exec$UNDERLINE$datasetSize$UNDERLINE$miniBatchSize$UNDERLINE$numThreads$UNDERLINE
			
			./$exec -d $datasetPath -i $inputDim -h $hiddenDim -m $miniBatchSize -n $numThreads -v 1 > $LOG_FILE
			cat $LOG_FILE
			result_path1=$result_path
			result_path1+=1
			mv ./$START_LOG $result_path1$UNDERLINE$START_LOG
			mv ./$END_LOG $result_path1$UNDERLINE$END_LOG
			mv ./$TIME_LOG $result_path1$UNDERLINE$TIME_LOG
			mv ./$LOG_FILE $result_path1$UNDERLINE$LOG_FILE
			
			./$exec -d $datasetPath -i $inputDim -h $hiddenDim -m $miniBatchSize -n $numThreads -v 2 > $LOG_FILE
			cat $LOG_FILE
			result_path2=$result_path
			result_path2+=2	
			mv ./$START_LOG $result_path2$UNDERLINE$START_LOG
			mv ./$END_LOG $result_path2$UNDERLINE$END_LOG
			mv ./$TIME_LOG $result_path2$UNDERLINE$TIME_LOG
			mv ./$LOG_FILE $result_path2$UNDERLINE$LOG_FILE
				
			./$exec -d $datasetPath -i $inputDim -h $hiddenDim -m $miniBatchSize -n $numThreads -v 3 > $LOG_FILE
			cat $LOG_FILE
			result_path3=$result_path
			result_path3+=3	
			mv ./$START_LOG $result_path3$UNDERLINE$START_LOG
			mv ./$END_LOG $result_path3$UNDERLINE$END_LOG
			mv ./$TIME_LOG $result_path3$UNDERLINE$TIME_LOG
			mv ./$LOG_FILE $result_path3$UNDERLINE$LOG_FILE
		
			./$exec -d $datasetPath -i $inputDim -h $hiddenDim -m $miniBatchSize -n $numThreads -v 4 > $LOG_FILE
			cat $LOG_FILE
			result_path4=$result_path
			result_path4+=4	
			mv ./$START_LOG $result_path4$UNDERLINE$START_LOG
			mv ./$END_LOG $result_path4$UNDERLINE$END_LOG
			mv ./$TIME_LOG $result_path4$UNDERLINE$TIME_LOG
			mv ./$LOG_FILE $result_path4$UNDERLINE$LOG_FILE
				
			./$exec -d $datasetPath -i $inputDim -h $hiddenDim -m $miniBatchSize -n $numThreads -v 5 > $LOG_FILE
			cat $LOG_FILE
			result_path5=$result_path
			result_path5+=5	
			mv ./$START_LOG $result_path5$UNDERLINE$START_LOG
			mv ./$END_LOG $result_path5$UNDERLINE$END_LOG
			mv ./$TIME_LOG $result_path5$UNDERLINE$TIME_LOG
			mv ./$LOG_FILE $result_path5$UNDERLINE$LOG_FILE
				
		done
	done
done

echo END!

