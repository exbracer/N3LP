#########################################################################
# File Name: new_run.sh
# Author: qiao_yuchen
# mail: qiaoyc14@mails.tsinghua.edu.cn
# Created Time: Wed Jul 20 20:02:19 2016
#########################################################################
#!/bin/bash

# input & output file location
DATASET_LOCATION=../n3lp_data/
RESULT_LOCATION=./new_result/

# parameters
INPUT_DIMENTION=512
HIDDEN_DIMENTION=512

DATASET_SIZE=$1
NUM_THREADS=$2
MINIBATCH_SIZE=128

# perf
perfFlag=$4
PERF=
PERF_FLAG=
if [ $perfFlag -ne 0 ]
then
	#PERF+=sudo\ perf\ record\ 
	#PERF+=-e\ probe_libtcmalloc:tc_malloc\ -agR\ 
	PERF+=perf\ stat\ 
	PERF+=-d\ 
	#PERF_FLAG+=-e\ 
	#PERF_FLAG+=branches,branch-misses,
	#PERF_FLAG+=cache-references,cache-misses,
	#PERF_FLAG+=L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,
	#PERF_FLAG+=L1-icache-loads,L1-icache-load-misses,
	#PERF_FLAG+=LLC-loads,LLC-load-misses,LLC-stores
	PERF_FALG+=\ -- 
	#PERF_FLAG+=cache-misses
	#PERF_FLAG+=--
fi

# numa
NUMA=
NUMCTL=
NUMAFLAG=
numaFlag=$3
if [ $numaFlag -ne 0 ]
then
	NUMA+=numa_
	NUMACTL+=numactl
	NUMAFLAG=--interleave=all\ 
	if [ $numaFlag -eq 1 ]
	then
		NUMAFLAG+=--physcpubind=16-31
	elif [ $numaFlag -eq 2 ]
	then
		NUMAFLAG+=--physcpubind=32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62
	elif [ $numaFlag -eq 3 ]
	then
		NUMAFLAG+=--physcpubind=8-11,24-27,40-43,56-59
	elif [ $numaFlag -eq 4 ]
	then
		NUMAFLAG+=--physcpubind=32-39,48-55
	elif [ $numaFlag -eq 5 ]
	then
		NUMAFLAG+=--physcpubind=8,10,12,14,24,26,28,30,40,42,44,46,56,58,60,62
	elif [ $numaFlag -eq 6 ]
	then
		NUMAFLAG+=--physcpubind=2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62
	elif [ $numaFlag -eq 7 ]
	then
		NUMAFLAG+=--physcpubind=0,4,8,12,17,21,25,29,34,38,42,46,51,55,59,63
	elif [ $numaFlag -eq 8 ]
	then
		NUMAFLAG+=--physcpubind=0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62
	fi
fi

# log file and statistc file
LOG_FILE=
LOG_FILE+=$RESULT_LOCATION
LOG_FILE+=runtime.log

STATISTIC_FILE=
STATISTIC_FILE+=$1
STATISTIC_FILE+=_result.stats

# executable
EXECUTABLE_FILE=(
	n3lp
	n3lp_tc
)

# const
UNDERLINE=_

dataset_path=$DATASET_LOCATION
dataset_path+=$DATASET_SIZE
inputDim=$INPUT_DIMENTION
hiddenDim=$HIDDEN_DIMENTION
miniBatchSize=$MINIBATCH_SIZE
numThreads=$NUM_THREADS

echo START!!

for elem in ${EXECUTABLE_FILE[*]}
do
	echo "" > $STATISTIC_FILE
	echo $elem

	#$PERF $PERF_FLAG $NUMACTL $NUMAFLAG 
	$NUMACTL $NUMAFLAG $PERF $PERF_FLAG ./$elem -d $dataset_path -i $inputDim -h $hiddenDim -m $miniBatchSize -n $numThreads #> $LOG_FILE
	cat $LOG_FILE
	cat $LOG_FILE | grep -n 'ms' | sed 's/\(.*\): \(.*\) ms./\2/g' >> $STATISTIC_FILE
	echo "" >> $STATISTIC_FILE
	mv $STATISTIC_FILE $RESULT_LOCATION$NUMA$elem$UNDERLINE$numThreads$UNDERLINE$STATISTIC_FILE

done
