#########################################################################
# File Name: run.sh
# Author: qiao_yuchen
# mail: qiaoyc14@mails.tsinghua.edu.cn
# Created Time: Fri Jul  8 09:23:52 2016
#########################################################################
#!/bin/bash

DATASET_LOCATION=../n3lp_data/
# argument 1 means the dataset location
DATASET_LOCATION+=$1

RESULT_LOCATION=./result/

INPUT_DIMENTION=50
HIDDEN_DIMENTION=50
MIN_I=0
MAX_I=0
MIN_J=2
MAX_J=2

LOG_FILE=$RESULT_LOCATION
LOG_FILE+=runtime.log

STATIC_FILE=_
STATIC_FILE+=$1
STATIC_FILE+=_result.static

EXECUTABLE_FILE=(
	n3lp
	n3lp_tc
)


echo START!!!

for elem in ${EXECUTABLE_FILE[*]}
do
	echo $elem
	echo "" > $STATIC_FILE
	for ((i=$MIN_I;i<=$MAX_I;i++))
	do
		for ((j=$MIN_J;j<=$MAX_J;j++))
		do
			miniBatchSize=$[128*2**$i]
			numThreads=$[2**$j]
			./$elem -d $DATASET_LOCATION -i $INPUT_DIMENTION -h $HIDDEN_DIMENTION -m $miniBatchSize -n $numThreads > $LOG_FILE 
			cat $LOG_FILE
			cat $LOG_FILE | sed -n '7,8p' >> $STATIC_FILE
			cat $LOG_FILE | grep -n 'ms' | sed 's/\(.*\): \(.*\) ms./\2/g' >> $STATIC_FILE
			echo "" >> $STATIC_FILE
		done
	done
	mv $STATIC_FILE $RESULT_LOCATION$elem$STATIC_FILE
done


